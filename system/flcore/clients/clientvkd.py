"""
flcore/clients/clientvkd.py — FedVKD 客户端（Data-Aware + Vulnerability-Aware Selective KD）
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
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
        self.global_rounds = args.global_rounds

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.teacher_model = None

        self.alpha_0 = args.alpha_0
        self.T_kd = args.temperature_kd
        self.gamma = args.gamma_schedule
        self.beta = args.beta_vkd
        self.ema_mu = args.ema_mu
        self.warmup_rounds = args.warmup_rounds
        self.vuln_threshold = args.vuln_threshold

        self.distill_weights = None
        self.class_vulnerability_ema = torch.zeros(self.num_classes, device=self.device)
        self.last_round_metrics = {}

    def set_distill_weights(self, local_class_counts, global_avg_per_class):
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

    def train(self, data_this_client, round_idx):
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )
        start_time = time.time()
        self.model.train()

        max_local_epochs = self.local_epochs
        can_estimate_vulnerability = (
            self.distill_weights is not None
            and self.distill_weights.sum() > 1e-8
            and self.teacher_model is not None
        )
        use_distill = round_idx >= self.warmup_rounds and can_estimate_vulnerability

        all_total_losses = []
        all_ce_losses = []
        all_kd_losses = []
        all_logit_kd_losses = []
        all_feat_kd_losses = []
        last_alpha = 0.0
        last_effective_weights = None

        for epoch in range(max_local_epochs):
            alpha_e = self._schedule_alpha(round_idx, epoch, max_local_epochs) if use_distill else 0.0
            last_alpha = alpha_e
            vulnerability_sum = torch.zeros(self.num_classes, device=self.device)
            vulnerability_batches = 0

            for x, y in data_this_client:
                if isinstance(x, list):
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device).long()

                self.optimizer.zero_grad()
                feat_local, logit_local = self._forward_with_feature(self.model, x)
                loss_ce = self.loss(logit_local, y)
                loss = loss_ce
                kd_value = 0.0
                logit_kd_value = 0.0
                feat_kd_value = 0.0
                feat_global = None
                logit_global = None

                if can_estimate_vulnerability:
                    with torch.no_grad():
                        feat_global, logit_global = self._forward_with_feature(self.teacher_model, x)
                        delta_batch = F.softmax(logit_global.detach(), dim=1) - F.softmax(logit_local.detach(), dim=1)
                        vulnerability_sum += torch.clamp(delta_batch, min=0.0).mean(dim=0)
                        vulnerability_batches += 1

                if use_distill and alpha_e > 0:
                    effective_weights = self._effective_distill_weights()
                    last_effective_weights = effective_weights
                    loss_logit = self._logit_kd_loss(logit_local, logit_global, effective_weights)
                    loss_feat = self._feature_alignment_loss(
                        feat_local, feat_global, logit_global, effective_weights
                    )
                    loss_kd = self.beta * loss_logit + (1 - self.beta) * loss_feat
                    loss = loss_ce + alpha_e * loss_kd
                    kd_value = (alpha_e * loss_kd).item()
                    logit_kd_value = loss_logit.item()
                    feat_kd_value = loss_feat.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

                all_total_losses.append(loss.item())
                all_ce_losses.append(loss_ce.item())
                all_kd_losses.append(kd_value)
                all_logit_kd_losses.append(logit_kd_value)
                all_feat_kd_losses.append(feat_kd_value)

            self._update_vulnerability(vulnerability_sum, vulnerability_batches)

        dw = self.distill_weights
        if use_distill and last_effective_weights is not None:
            effective_weights = last_effective_weights
        else:
            effective_weights = None
        self.last_round_metrics = {
            "total_loss": float(np.mean(all_total_losses)) if all_total_losses else 0.0,
            "ce_loss": float(np.mean(all_ce_losses)) if all_ce_losses else 0.0,
            "kd_loss": float(np.mean(all_kd_losses)) if all_kd_losses else 0.0,
            "logit_kd_loss": float(np.mean(all_logit_kd_losses)) if all_logit_kd_losses else 0.0,
            "feat_kd_loss": float(np.mean(all_feat_kd_losses)) if all_feat_kd_losses else 0.0,
            "alpha": float(last_alpha),
            "use_distill": float(use_distill),
            "num_distill_classes": int((dw > 0).sum().item()) if dw is not None else 0,
            "num_effective_distill_classes": int((effective_weights > 0).sum().item()) if effective_weights is not None else 0,
            "num_vulnerable_classes": int((self.class_vulnerability_ema >= self.vuln_threshold).sum().item()),
            "distill_weight_mean": float(dw.mean().item()) if dw is not None else 0.0,
            "effective_weight_mean": float(effective_weights.mean().item()) if effective_weights is not None else 0.0,
            "effective_weight_max": float(effective_weights.max().item()) if effective_weights is not None else 0.0,
            "vulnerability_mean": float(self.class_vulnerability_ema.mean().item()),
            "vulnerability_max": float(self.class_vulnerability_ema.max().item()),
        }

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _update_vulnerability(self, vulnerability_sum, vulnerability_batches):
        if vulnerability_batches <= 0:
            return
        vulnerability_batch = vulnerability_sum / max(vulnerability_batches, 1)
        self.class_vulnerability_ema = (
            self.ema_mu * self.class_vulnerability_ema
            + (1.0 - self.ema_mu) * vulnerability_batch
        )

    def _effective_distill_weights(self):
        if self.distill_weights is None:
            return None

        if self.distill_weights.sum() < 1e-8:
            return torch.zeros_like(self.distill_weights)

        vuln = self.class_vulnerability_ema.detach()
        if vuln.max() > 1e-8:
            vuln_norm = vuln / (vuln.max() + 1e-8)
        else:
            vuln_norm = torch.zeros_like(vuln)

        above_threshold = (vuln >= self.vuln_threshold).float()
        soft_factor = (
            (1.0 - above_threshold) * (0.25 + 0.25 * vuln_norm)
            + above_threshold * (0.50 + 0.50 * vuln_norm)
        )

        effective = self.distill_weights * soft_factor
        if effective.sum() < 1e-8 and self.distill_weights.sum() > 1e-8:
            effective = 0.25 * self.distill_weights
        return effective

    def _logit_kd_loss(self, logit_local, logit_global, effective_weights):
        T = self.T_kd
        if effective_weights is None:
            return torch.tensor(0.0, device=logit_local.device)
        weight_sum = effective_weights.sum()
        if weight_sum < 1e-8:
            return torch.tensor(0.0, device=logit_local.device)

        weight_norm = effective_weights / weight_sum
        log_p_student = F.log_softmax(logit_local / T, dim=1)
        p_teacher = F.softmax(logit_global / T, dim=1)
        kl_per_class = F.kl_div(log_p_student, p_teacher, reduction="none")
        weighted_kl = (kl_per_class * weight_norm.unsqueeze(0)).sum(dim=1).mean()
        return weighted_kl * (T ** 2)

    def _feature_alignment_loss(self, feat_local, feat_global, logit_global, effective_weights):
        if effective_weights is None or effective_weights.sum() < 1e-8:
            return torch.tensor(0.0, device=feat_local.device)

        prob_global = F.softmax(logit_global.detach(), dim=1)
        weight_norm = effective_weights / (effective_weights.sum() + 1e-8)
        sample_weight = (prob_global * weight_norm.unsqueeze(0)).sum(dim=1)
        feat_local_norm = F.normalize(feat_local, dim=1)
        feat_global_norm = F.normalize(feat_global.detach(), dim=1)
        normalized_feature_distance = ((feat_local_norm - feat_global_norm) ** 2).sum(dim=1)
        return (sample_weight * normalized_feature_distance).sum() / (sample_weight.sum() + 1e-8)

    def _schedule_alpha(self, round_idx, epoch, total_epochs):
        if round_idx < self.warmup_rounds:
            return 0.0
        total_kd_rounds = max(self.global_rounds - self.warmup_rounds, 1)
        round_progress = min(
            1.0,
            max(0.0, (round_idx + 1 - self.warmup_rounds) / total_kd_rounds),
        )
        epoch_progress = (epoch + 1) / max(total_epochs, 1)
        return self.alpha_0 * (round_progress ** self.gamma) * epoch_progress

    def _forward_with_feature(self, model, x):
        feature = model.base(x)
        if feature.dim() > 2:
            feature = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        logit = model.head(feature)
        return feature, logit

    def set_parameters(self, model):
        self.model.load_state_dict(model.state_dict())
        self.teacher_model = copy.deepcopy(model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
