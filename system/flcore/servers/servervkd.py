import torch
import torch.nn as nn
import numpy as np
import time
import copy
import torch.nn.functional as F

class clientVKD(object):
    def __init__(self, args, id, train_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.weight_decay = args.weight_decay

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.teacher_model = None

        # ====== FedVKD 超参数 ======
        self.alpha_0 = args.alpha_0          # 基础蒸馏强度，推荐 0.5
        self.T_kd = args.temperature_kd      # 蒸馏温度，推荐 3.0
        self.gamma = args.gamma_schedule     # 调度曲线指数，推荐 1.5
        self.beta = args.beta_vkd            # logit vs feature 权重，推荐 0.7
        self.ema_mu = args.ema_mu            # EMA 平滑系数，推荐 0.5

        # ★ warmup 轮数：前若干轮纯 CE，让全局模型学到有意义的初始知识
        self.warmup_rounds = getattr(args, 'warmup_rounds', 10)

        # ★ 脆弱度激活的绝对阈值（概率差 5% 才认为是有意义的退化）
        self.vuln_threshold = 0.05

        # 脆弱度状态（跨 round 保持，但 warmup 期间不累积）
        self.vulnerability = None

    def train(self, data_this_client, round):
        """FedVKD 客户端本地训练主函数"""
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay
        )
        start_time = time.time()
        trainloader = data_this_client
        self.model.train()

        print(f"\n-------------client: {self.id}-------------")

        max_local_epochs = self.local_epochs

        # ★ warmup：前 warmup_rounds 轮纯 CE，避免蒸馏随机模型
        use_distill = (round >= self.warmup_rounds)

        all_ce_losses = []
        all_kd_losses = []
        all_total_losses = []
        last_alpha = 0.0

        for epoch in range(max_local_epochs):
            # ====== Step 1: 计算脆弱度（仅在启用蒸馏时） ======
            if use_distill:
                vuln = self._compute_vulnerability(trainloader, epoch)
                alpha_e = self._schedule_alpha(epoch, max_local_epochs)
            else:
                vuln = torch.zeros(self.num_classes).to(self.device)
                alpha_e = 0.0
            last_alpha = alpha_e

            # ====== Step 2: 训练 ======
            epoch_loss_collector = []
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device).long()

                self.optimizer.zero_grad()

                if use_distill:
                    feat_local, logit_local = self._forward_with_feature(self.model, x)
                    with torch.no_grad():
                        feat_global, logit_global = self._forward_with_feature(self.teacher_model, x)

                    loss_ce = self.loss(logit_local, y)
                    loss_logit = self._logit_kd_loss(logit_local, logit_global, vuln)
                    loss_feat = self._feature_alignment_loss(
                        feat_local, feat_global, logit_global, vuln
                    )
                    loss_kd = self.beta * loss_logit + (1 - self.beta) * loss_feat
                    loss = loss_ce + alpha_e * loss_kd
                    kd_value = (alpha_e * loss_kd).item()
                else:
                    # warmup 阶段：纯 CE
                    _, logit_local = self._forward_with_feature(self.model, x)
                    loss_ce = self.loss(logit_local, y)
                    loss = loss_ce
                    kd_value = 0.0

                loss.backward()
                # ★ 加梯度裁剪，防止任何阶段梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

                epoch_loss_collector.append(loss.item())
                all_total_losses.append(loss.item())
                all_ce_losses.append(loss_ce.item())
                all_kd_losses.append(kd_value)

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            print('Epoch: %d Loss: %f alpha: %.4f distill: %s' % (
                epoch, epoch_loss, alpha_e, str(use_distill)
            ))

        # ★ 记录本轮指标
        self.last_round_metrics = {
            "total_loss": sum(all_total_losses) / len(all_total_losses) if all_total_losses else 0.0,
            "ce_loss": sum(all_ce_losses) / len(all_ce_losses) if all_ce_losses else 0.0,
            "kd_loss": sum(all_kd_losses) / len(all_kd_losses) if all_kd_losses else 0.0,
            "alpha": last_alpha,
            "use_distill": float(use_distill),
            "vuln_mean": self.vulnerability.mean().item() if self.vulnerability is not None else 0.0,
            "vuln_max": self.vulnerability.max().item() if self.vulnerability is not None else 0.0,
            "num_vuln_classes": int((self.vulnerability > 0.1).sum().item()) if self.vulnerability is not None else 0,
        }

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # ================================================================
    #                    模块一：在线脆弱度检测 (OVD)
    # ================================================================

    @torch.no_grad()
    def _compute_vulnerability(self, trainloader, epoch):
        """
        在线脆弱度检测（修正版）
        
        修正点：
        1. 用绝对阈值代替强制归一化，避免小信号被放大成伪信号
        2. 整体退化太小时直接置零
        3. EMA 平滑系数变小（0.5），让信号能快速更新
        """
        self.model.eval()
        self.teacher_model.eval()

        delta = torch.zeros(self.num_classes).to(self.device)
        sample_count = 0

        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)

            prob_global = F.softmax(self.teacher_model(x), dim=1)
            prob_local = F.softmax(self.model(x), dim=1)
            delta += (prob_global - prob_local).sum(dim=0)
            sample_count += x.size(0)

        delta = delta / max(sample_count, 1)
        delta = torch.clamp(delta, min=0)

        # ★ 关键修正：用绝对阈值
        # delta_c > threshold 才算严重脆弱（vuln=1）
        # 整体最大值都很小时，认为没有脆弱类，直接置零
        if delta.max() < self.vuln_threshold * 0.5:
            vuln_raw = torch.zeros_like(delta)
        else:
            vuln_raw = torch.clamp(delta / self.vuln_threshold, max=1.0)

        # EMA 平滑
        if self.vulnerability is None:
            self.vulnerability = vuln_raw
        else:
            self.vulnerability = (
                self.ema_mu * self.vulnerability + (1 - self.ema_mu) * vuln_raw
            )

        self.model.train()
        return self.vulnerability

    # ================================================================
    #              模块二：Logit-Level 自适应蒸馏（修正版）
    # ================================================================

    def _logit_kd_loss(self, logit_local, logit_global, vulnerability):
        """
        修正点：
        1. vulnerability 归一化为权重和=1（防止总量级爆炸）
        2. 去掉 T² 缩放（vulnerability 已经在控制权重，不需要再放大）
        """
        T = self.T_kd
        p_teacher = F.softmax(logit_global / T, dim=1)
        log_p_student = F.log_softmax(logit_local / T, dim=1)
        log_p_teacher = torch.log(p_teacher + 1e-8)

        kl_per_class = p_teacher * (log_p_teacher - log_p_student)  # [B, C]

        # ★ 修正：归一化 vulnerability，保证总权重为 1
        vuln_sum = vulnerability.sum()
        if vuln_sum < 1e-6:
            return torch.tensor(0.0, device=logit_local.device)
        weight = vulnerability / vuln_sum  # [C]，sum=1

        weighted_kl = (kl_per_class * weight.unsqueeze(0)).sum(dim=1).mean()
        return weighted_kl  # ★ 去掉 T²

    # ================================================================
    #              模块三：Feature-Level 结构保护（修正版）
    # ================================================================

    def _feature_alignment_loss(self, feat_local, feat_global, logit_global, vulnerability):
        """
        修正点：vulnerability 归一化，避免量级失控
        """
        # ★ 整体没有脆弱类时直接返回 0
        if vulnerability.sum() < 1e-6:
            return torch.tensor(0.0, device=feat_local.device)

        prob_global = F.softmax(logit_global, dim=1)  # [B, C]
        # 归一化的 vulnerability
        vuln_norm = vulnerability / (vulnerability.sum() + 1e-8)
        sample_weight = (prob_global * vuln_norm.unsqueeze(0)).sum(dim=1)  # [B]

        feat_local_norm = F.normalize(feat_local, dim=1)
        feat_global_norm = F.normalize(feat_global, dim=1)

        mse_per_sample = ((feat_local_norm - feat_global_norm) ** 2).sum(dim=1)  # [B]
        loss = (sample_weight * mse_per_sample).sum() / (sample_weight.sum() + 1e-8)
        return loss

    # ================================================================
    #              模块四：Epoch-wise 渐进调度
    # ================================================================

    def _schedule_alpha(self, epoch, total_epochs):
        progress = (epoch + 1) / total_epochs
        alpha_e = self.alpha_0 * (progress ** self.gamma)
        return alpha_e

    # ================================================================
    #              辅助函数
    # ================================================================

    def _forward_with_feature(self, model, x):
        feature = model.base(x)
        logit = model.head(feature)
        return feature, logit

    def set_parameters(self, model):
        """从服务器接收全局模型参数"""
        global_w = model.state_dict()
        self.model.load_state_dict(global_w)
        self.teacher_model = copy.deepcopy(model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False