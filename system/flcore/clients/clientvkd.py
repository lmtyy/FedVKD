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
        self.alpha_0 = args.alpha_0          # 基础蒸馏强度，推荐 1.0
        self.T_kd = args.temperature_kd      # 蒸馏温度，推荐 3.0
        self.gamma = args.gamma_schedule     # 调度曲线指数，推荐 1.5
        self.beta = args.beta_vkd            # logit vs feature 权重，推荐 0.7
        self.ema_mu = args.ema_mu            # EMA 平滑系数，推荐 0.9

        # 脆弱度状态（跨 round 保持）
        self.vulnerability = None  # shape [C]，延迟初始化

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

        # ★ 累积整轮的 CE 和 KD loss（用于 wandb 上报）
        all_ce_losses = []
        all_kd_losses = []
        all_total_losses = []

        for epoch in range(max_local_epochs):
            # ====== Step 1: 每个 epoch 开始时计算脆弱度 ======
            vuln = self._compute_vulnerability(trainloader, epoch)

            # ====== Step 2: 计算渐进调度系数 ======
            alpha_e = self._schedule_alpha(epoch, max_local_epochs)

            # ====== Step 3: 正常训练 + 蒸馏 ======
            epoch_loss_collector = []
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                y = y.long()

                self.optimizer.zero_grad()

                # 本地模型前向（需要同时拿 feature 和 logit）
                feat_local, logit_local = self._forward_with_feature(self.model, x)

                # 全局模型前向（teacher，不需要梯度）
                with torch.no_grad():
                    feat_global, logit_global = self._forward_with_feature(self.teacher_model, x)

                # 分类损失
                loss_ce = self.loss(logit_local, y)

                # Logit-level 自适应蒸馏损失
                loss_logit = self._logit_kd_loss(logit_local, logit_global, vuln)

                # Feature-level 结构保护损失
                loss_feat = self._feature_alignment_loss(
                    feat_local, feat_global, logit_global, vuln
                )

                # 总损失
                loss_kd = self.beta * loss_logit + (1 - self.beta) * loss_feat
                loss = loss_ce + alpha_e * loss_kd

                loss.backward()
                self.optimizer.step()

                epoch_loss_collector.append(loss.item())

                # ★ 分别累积（用于 wandb）
                all_total_losses.append(loss.item())
                all_ce_losses.append(loss_ce.item())
                all_kd_losses.append((alpha_e * loss_kd).item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            print('Epoch: %d Loss: %f alpha: %.4f' % (epoch, epoch_loss, alpha_e))

        # ★ 记录本轮指标，供 server 的 _collect_client_metrics() 读取
        self.last_round_metrics = {
            "total_loss": sum(all_total_losses) / len(all_total_losses) if all_total_losses else 0.0,
            "ce_loss": sum(all_ce_losses) / len(all_ce_losses) if all_ce_losses else 0.0,
            "kd_loss": sum(all_kd_losses) / len(all_kd_losses) if all_kd_losses else 0.0,
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
        在线脆弱度检测 (Online Vulnerability Detection)
        
        核心思想：比较全局模型和当前本地模型对各类别的平均预测概率差。
        如果本地模型对某类的预测概率比全局模型低很多，说明该类知识正在被遗忘。
        
        返回: vulnerability tensor, shape [C], 值域 [0, 1]
        """
        self.model.eval()
        self.teacher_model.eval()

        # 累积每个类别的概率差
        delta = torch.zeros(self.num_classes).to(self.device)
        sample_count = 0

        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)

            # 全局模型预测概率
            logit_global = self.teacher_model(x)
            prob_global = F.softmax(logit_global, dim=1)  # [B, C]

            # 本地模型预测概率
            logit_local = self.model(x)
            prob_local = F.softmax(logit_local, dim=1)  # [B, C]

            # 概率差：全局 - 本地（正值表示本地在该类上退化）
            diff = prob_global - prob_local  # [B, C]
            delta += diff.sum(dim=0)  # [C]
            sample_count += x.size(0)

        # 平均概率差
        delta = delta / sample_count  # [C]

        # 只保留正值（本地退化的类别）
        delta = torch.clamp(delta, min=0)

        # 归一化到 [0, 1]
        max_val = delta.max()
        if max_val > 1e-8:
            vuln_raw = delta / max_val
        else:
            vuln_raw = torch.zeros(self.num_classes).to(self.device)

        # EMA 平滑（跨 epoch 和跨 round 稳定信号）
        if self.vulnerability is None:
            self.vulnerability = vuln_raw
        else:
            self.vulnerability = (
                self.ema_mu * self.vulnerability + (1 - self.ema_mu) * vuln_raw
            )

        self.model.train()
        return self.vulnerability

    # ================================================================
    #              模块二：Logit-Level 自适应蒸馏
    # ================================================================

    def _logit_kd_loss(self, logit_local, logit_global, vulnerability):
        """
        脆弱度加权的 Logit KD 损失
        
        与 FedVLS 的区别：
        - FedVLS 只对空类做 KL（二值）
        - 我们对所有类做加权 KL（连续权重）
        
        Args:
            logit_local: 本地模型输出 [B, C]
            logit_global: 全局模型输出 [B, C]
            vulnerability: 脆弱度分数 [C]
        
        Returns:
            加权 KL 散度损失（标量）
        """
        T = self.T_kd

        # soft target
        p_teacher = F.softmax(logit_global / T, dim=1)  # [B, C]
        log_p_student = F.log_softmax(logit_local / T, dim=1)  # [B, C]

        # 逐类 KL 散度: [B, C]
        kl_per_class = p_teacher * (torch.log(p_teacher + 1e-8) - log_p_student)

        # 用脆弱度加权：对脆弱类施加更强的蒸馏
        # vulnerability shape [C] -> broadcast to [1, C]
        weight = vulnerability.unsqueeze(0)  # [1, C]

        # 加权求和
        weighted_kl = (kl_per_class * weight).sum(dim=1).mean()  # 先对类求和，再对 batch 求均值

        # 温度缩放
        loss = weighted_kl * (T ** 2)

        return loss

    # ================================================================
    #              模块三：Feature-Level 结构保护
    # ================================================================

    def _feature_alignment_loss(self, feat_local, feat_global, logit_global, vulnerability):
        """
        脆弱度加权的 Feature Alignment 损失
        
        核心思想：对于全局模型认为属于脆弱类的样本，强制本地模型的特征表示
        与全局模型对齐，防止特征空间结构坍塌。
        
        使用简化版本：样本权重 = Σ_c v_c * p_global(c|x)
        
        Args:
            feat_local: 本地模型特征 [B, D]
            feat_global: 全局模型特征 [B, D]
            logit_global: 全局模型 logit [B, C]（用于计算样本权重）
            vulnerability: 脆弱度分数 [C]
        
        Returns:
            加权 MSE 损失（标量）
        """
        # 全局模型对各类的预测概率
        prob_global = F.softmax(logit_global, dim=1)  # [B, C]

        # 样本权重：如果全局模型认为该样本属于脆弱类，则权重高
        # w(x) = Σ_c v_c * p_global(c|x)
        sample_weight = (prob_global * vulnerability.unsqueeze(0)).sum(dim=1)  # [B]

        # 归一化特征（防止尺度问题）
        feat_local_norm = F.normalize(feat_local, dim=1)
        feat_global_norm = F.normalize(feat_global, dim=1)

        # 逐样本 MSE
        mse_per_sample = ((feat_local_norm - feat_global_norm) ** 2).sum(dim=1)  # [B]

        # 加权平均
        loss = (sample_weight * mse_per_sample).mean()

        return loss

    # ================================================================
    #              模块四：Epoch-wise 渐进调度
    # ================================================================

    def _schedule_alpha(self, epoch, total_epochs):
        """
        渐进调度：训练初期保护弱，后期保护强
        
        α(e) = α_0 * ((e+1) / E)^γ
        
        γ=1: 线性增长
        γ>1: 凸增长（前期温和，后期加强）—— 推荐 1.5
        γ<1: 凹增长（前期就较强）
        """
        progress = (epoch + 1) / total_epochs
        alpha_e = self.alpha_0 * (progress ** self.gamma)
        return alpha_e

    # ================================================================
    #              辅助函数：提取 feature + logit
    # ================================================================

    def _forward_with_feature(self, model, x):
        """
        同时获取模型的 feature（倒数第二层输出）和 logit（最终输出）
        
        适配 BaseHeadSplit 结构：
            model.base = feature extractor (backbone without fc)
            model.head = classifier (fc layer)
        
        Returns:
            feature: [B, D] 特征向量
            logit: [B, C] 分类 logit
        """
        # BaseHeadSplit 结构: base -> head
        feature = model.base(x)   # [B, D]
        logit = model.head(feature)  # [B, C]
        return feature, logit

    # ================================================================
    #              参数设置（服务器调用）
    # ================================================================

    def set_parameters(self, model):
        """从服务器接收全局模型参数"""
        global_w = model.state_dict()
        self.model.load_state_dict(global_w)
        # 全局模型作为 teacher（冻结）
        self.teacher_model = copy.deepcopy(model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
