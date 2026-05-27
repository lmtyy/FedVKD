"""
flcore/clients/clientvkd.py — FedVKD 客户端（修复版 v2）

相对仓库现版的改动：
1. ★ vuln 计算时机修正：从「epoch 开头」移到「epoch 末尾」
   - 旧版：epoch 0 开头，本地模型刚 = 全局模型，vuln=0；
           local_epochs=1 时 EMA 衰减，vuln 永远接近 0，蒸馏失效。
   - 新版：用「上一轮存下来的 vuln」做本 epoch 蒸馏；
           本 epoch 训完再算新 vuln 存起来。
           local_epochs=1 也能正常工作。
2. 新增 last_round_metrics 中 num_distill_classes 字段，便于 wandb 看出哪些类被实际蒸馏。
"""
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
        self.alpha_0 = args.alpha_0
        self.T_kd = args.temperature_kd
        self.gamma = args.gamma_schedule
        self.beta = args.beta_vkd
        self.ema_mu = args.ema_mu

        # warmup 轮数 + 脆弱度阈值（可由 main.py 注入）
        self.warmup_rounds = getattr(args, 'warmup_rounds', 10)
        self.vuln_threshold = getattr(args, 'vuln_threshold', 0.05)

        # 脆弱度状态（跨 round 持久化）
        self.vulnerability = None
        self.last_round_metrics = {}

    # ================================================================
    #                          主训练循环
    # ================================================================
    def train(self, data_this_client, round_idx):
        """FedVKD 客户端本地训练"""
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay
        )
        start_time = time.time()
        trainloader = data_this_client
        self.model.train()
        #print(f"\n-------------client: {self.id}-------------")

        max_local_epochs = self.local_epochs
        use_distill = (round_idx >= self.warmup_rounds)

        all_ce_losses, all_kd_losses, all_total_losses = [], [], []
        last_alpha = 0.0

        for epoch in range(max_local_epochs):
            # ★ 关键修正：用「上一轮残留的 vulnerability」做本 epoch 的蒸馏
            #    避免 epoch 0 开头本地=全局导致 vuln=0、local_epochs=1 时蒸馏失效
            if use_distill and self.vulnerability is not None:
                vuln = self.vulnerability
                alpha_e = self._schedule_alpha(epoch, max_local_epochs)
            else:
                # 第一次进入蒸馏阶段时 vuln 为 None，此 epoch 仍纯 CE
                # 等本 epoch 训完会算出 vuln，下个 epoch / 下一轮就能用
                vuln = torch.zeros(self.num_classes, device=self.device)
                alpha_e = 0.0
            last_alpha = alpha_e
            do_kd_this_epoch = (alpha_e > 0)

            epoch_loss_collector = []
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device).long()

                self.optimizer.zero_grad()

                if do_kd_this_epoch:
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
                    kd_value = (alpha_e * loss_kd).item() if torch.is_tensor(loss_kd) else 0.0
                else:
                    _, logit_local = self._forward_with_feature(self.model, x)
                    loss_ce = self.loss(logit_local, y)
                    loss = loss_ce
                    kd_value = 0.0

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

                epoch_loss_collector.append(loss.item())
                all_total_losses.append(loss.item())
                all_ce_losses.append(loss_ce.item())
                all_kd_losses.append(kd_value)

            # ★ 关键修正：epoch 训完之后才更新 vulnerability
            #    使用「本 epoch 训练后的本地模型」与「全局模型」对比，
            #    才能反映本 epoch 中本地真正退化了哪些类。
            if use_distill:
                self._update_vulnerability(trainloader)

            epoch_loss = sum(epoch_loss_collector) / max(len(epoch_loss_collector), 1)
            #print('Epoch: %d Loss: %f alpha: %.4f distill: %s' % (
            #   epoch, epoch_loss, alpha_e, str(do_kd_this_epoch)))

        # ====== 记录指标供 server 上报 wandb ======
        self.last_round_metrics = {
            "total_loss": float(np.mean(all_total_losses)) if all_total_losses else 0.0,
            "ce_loss":    float(np.mean(all_ce_losses))    if all_ce_losses    else 0.0,
            "kd_loss":    float(np.mean(all_kd_losses))    if all_kd_losses    else 0.0,
            "alpha": float(last_alpha),
            "use_distill": float(use_distill),
            "vuln_mean": self.vulnerability.mean().item() if self.vulnerability is not None else 0.0,
            "vuln_max":  self.vulnerability.max().item()  if self.vulnerability is not None else 0.0,
            "num_vuln_classes":
                int((self.vulnerability > 0.1).sum().item()) if self.vulnerability is not None else 0,
            "num_distill_classes":
                int((self.vulnerability > 0.5).sum().item()) if self.vulnerability is not None else 0,
        }

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # ================================================================
    #            模块一：在线脆弱度检测（OVD），改为 epoch 末更新
    # ================================================================
    @torch.no_grad()
    def _update_vulnerability(self, trainloader):
        """epoch 末调用：测量本地模型与全局模型在各类上的概率差，
        用绝对阈值激活，EMA 平滑。"""
        was_training = self.model.training
        self.model.eval()
        self.teacher_model.eval()

        delta = torch.zeros(self.num_classes, device=self.device)
        sample_count = 0
        for x, _ in trainloader:
            if isinstance(x, list):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            prob_global = F.softmax(self.teacher_model(x), dim=1)
            prob_local = F.softmax(self.model(x), dim=1)
            delta += (prob_global - prob_local).sum(dim=0)
            sample_count += x.size(0)

        delta = delta / max(sample_count, 1)
        delta = torch.clamp(delta, min=0)

        # 绝对阈值：整体退化太小时直接置零，避免被强制归一化放大成假信号
        if delta.max() < self.vuln_threshold * 0.5:
            vuln_raw = torch.zeros_like(delta)
        else:
            vuln_raw = torch.clamp(delta / self.vuln_threshold, max=1.0)

        # EMA 平滑（跨 epoch、跨 round 都稳）
        if self.vulnerability is None:
            self.vulnerability = vuln_raw
        else:
            self.vulnerability = (
                self.ema_mu * self.vulnerability + (1 - self.ema_mu) * vuln_raw
            )

        if was_training:
            self.model.train()

    # ================================================================
    #                  模块二：Logit-Level 蒸馏
    # ================================================================
    def _logit_kd_loss(self, logit_local, logit_global, vulnerability):
        """脆弱度加权 KL 散度。vulnerability 归一化为 sum=1，避免量级失控。"""
        T = self.T_kd
        p_teacher = F.softmax(logit_global / T, dim=1)
        log_p_student = F.log_softmax(logit_local / T, dim=1)
        log_p_teacher = torch.log(p_teacher + 1e-8)
        kl_per_class = p_teacher * (log_p_teacher - log_p_student)  # [B, C]

        vuln_sum = vulnerability.sum()
        if vuln_sum < 1e-6:
            return torch.tensor(0.0, device=logit_local.device)
        weight = vulnerability / vuln_sum  # [C], sum=1

        weighted_kl = (kl_per_class * weight.unsqueeze(0)).sum(dim=1).mean()
        return weighted_kl

    # ================================================================
    #                  模块三：Feature-Level 对齐
    # ================================================================
    def _feature_alignment_loss(self, feat_local, feat_global, logit_global, vulnerability):
        """对脆弱类样本，强制本地特征对齐全局特征。"""
        if vulnerability.sum() < 1e-6:
            return torch.tensor(0.0, device=feat_local.device)

        prob_global = F.softmax(logit_global, dim=1)  # [B, C]
        vuln_norm = vulnerability / (vulnerability.sum() + 1e-8)
        sample_weight = (prob_global * vuln_norm.unsqueeze(0)).sum(dim=1)  # [B]

        feat_local_norm = F.normalize(feat_local, dim=1)
        feat_global_norm = F.normalize(feat_global, dim=1)

        mse_per_sample = ((feat_local_norm - feat_global_norm) ** 2).sum(dim=1)
        loss = (sample_weight * mse_per_sample).sum() / (sample_weight.sum() + 1e-8)
        return loss

    # ================================================================
    #                    模块四：渐进调度
    # ================================================================
    def _schedule_alpha(self, epoch, total_epochs):
        """α(e) = α_0 * ((e+1)/E)^γ"""
        progress = (epoch + 1) / total_epochs
        return self.alpha_0 * (progress ** self.gamma)

    # ================================================================
    #                          辅助
    # ================================================================
    def _forward_with_feature(self, model, x):
        """适配 BaseHeadSplit 结构：base 提特征，head 出 logit。"""
        feature = model.base(x)
        logit = model.head(feature)
        return feature, logit

    def set_parameters(self, model):
        """从 server 接收全局模型；
        ★ 注意：set_parameters 不重置 vulnerability，让脆弱度跨 round 累积。"""
        self.model.load_state_dict(model.state_dict())
        self.teacher_model = copy.deepcopy(model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
