"""
flcore/clients/clientvkd.py — FedVKD 客户端（v3: Data-Aware Selective KD）
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

        # ====== FedVKD 超参数（v3 精简版）======
        self.alpha_0 = args.alpha_0
        self.T_kd = args.temperature_kd
        self.gamma = args.gamma_schedule
        self.beta = args.beta_vkd

        # ====== Data-Aware 蒸馏权重（延迟初始化）======
        self.distill_weights = None
        self.last_round_metrics = {}

    # ================================================================
    #          Data-Aware 权重计算（由 server 在创建 client 后调用）
    # ================================================================
    def set_distill_weights(self, local_class_counts, global_avg_per_class):
        """根据本地数据分布计算每个类的蒸馏权重。"""
        weights = torch.zeros(self.num_classes, device=self.device)
        for c in range(self.num_classes):
            n_c = local_class_counts[c]
            if n_c == 0:
                weights[c] = 1.0
            elif n_c < global_avg_per_class:
                weights[c] = max(0.0, 1.0 - n_c / global_avg_per_class)
            else:
                weights[c] = 0.0
        self.distill_weights = weights

    # ================================================================
    #                          主训练循环
    # ================================================================
    def train(self, data_this_client, round_idx):
        """FedVKD 客户端本地训练（Data-Aware Selective KD）"""
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay
        )
        start_time = time.time()
        trainloader = data_this_client
        self.model.train()

        max_local_epochs = self.local_epochs

        use_distill = (self.distill_weights is not None and
                       self.distill_weights.sum() > 1e-6 and
                       self.teacher_model is not None)

        all_ce_losses, all_kd_losses, all_total_losses = [], [], []
        last_alpha = 0.0

        for epoch in range(max_local_epochs):
            if use_distill:
                alpha_e = self._schedule_alpha(epoch, max_local_epochs)
            else:
                alpha_e = 0.0
            last_alpha = alpha_e
            do_kd_this_epoch = (alpha_e > 0)

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
                    loss_logit = self._logit_kd_loss(logit_local, logit_global)
                    loss_feat = self._feature_alignment_loss(
                        feat_local, feat_global, logit_global
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

                all_total_losses.append(loss.item())
                all_ce_losses.append(loss_ce.item())
                all_kd_losses.append(kd_value)

        # ====== 记录指标供 server 上报 ======
        dw = self.distill_weights
        self.last_round_metrics = {
            "total_loss": float(np.mean(all_total_losses)) if all_total_losses else 0.0,
            "ce_loss":    float(np.mean(all_ce_losses))    if all_ce_losses    else 0.0,
            "kd_loss":    float(np.mean(all_kd_losses))    if all_kd_losses    else 0.0,
            "alpha": float(last_alpha),
            "use_distill": float(use_distill),
            "num_distill_classes": int((dw > 0).sum().item()) if dw is not None else 0,
            "num_full_distill_classes": int((dw >= 0.99).sum().item()) if dw is not None else 0,
            "distill_weight_mean": float(dw.mean().item()) if dw is not None else 0.0,
        }

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # ================================================================
    #                  模块一：Logit-Level 蒸馏（Data-Aware 加权）
    # ================================================================
    def _logit_kd_loss(self, logit_local, logit_global):
        """Data-Aware 加权 KL 散度：只蒸馏缺失/不足的类。"""
        T = self.T_kd
        weights = self.distill_weights

        p_teacher = F.softmax(logit_global / T, dim=1)
        log_p_student = F.log_softmax(logit_local / T, dim=1)
        log_p_teacher = torch.log(p_teacher + 1e-8)
        kl_per_class = p_teacher * (log_p_teacher - log_p_student)

        weight_sum = weights.sum()
        if weight_sum < 1e-6:
            return torch.tensor(0.0, device=logit_local.device)
        weight_norm = weights / weight_sum

        weighted_kl = (kl_per_class * weight_norm.unsqueeze(0)).sum(dim=1).mean()
        return weighted_kl * (T ** 2)

    # ================================================================
    #                  模块二：Feature-Level 对齐（Data-Aware 加权）
    # ================================================================
    def _feature_alignment_loss(self, feat_local, feat_global, logit_global):
        """对缺失类样本，强制本地特征对齐全局特征。"""
        weights = self.distill_weights

        if weights.sum() < 1e-6:
            return torch.tensor(0.0, device=feat_local.device)

        prob_global = F.softmax(logit_global, dim=1)
        weight_norm = weights / (weights.sum() + 1e-8)
        sample_weight = (prob_global * weight_norm.unsqueeze(0)).sum(dim=1)

        feat_local_norm = F.normalize(feat_local, dim=1)
        feat_global_norm = F.normalize(feat_global, dim=1)

        mse_per_sample = ((feat_local_norm - feat_global_norm) ** 2).sum(dim=1)
        loss = (sample_weight * mse_per_sample).sum() / (sample_weight.sum() + 1e-8)
        return loss

    # ================================================================
    #                    模块三：渐进调度
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
        """从 server 接收全局模型。"""
        self.model.load_state_dict(model.state_dict())
        self.teacher_model = copy.deepcopy(model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False