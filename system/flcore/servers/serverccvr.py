"""
flcore/servers/serverccvr.py — CCVR 病根诊断（NeurIPS 2021）
诊断目的：判断 non-IID 下精度瓶颈到底在 classifier head 还是 feature。
流程：正常 FedAvg 训练 → 冻结 base → 用每类特征高斯采样虚拟特征 → 重训 head → 对比。
"""
import torch
import torch.nn as nn
import numpy as np
import copy
from torch.distributions.multivariate_normal import MultivariateNormal

from flcore.servers.serveravg import FedAvg

class CCVR(FedAvg):
    def __init__(self, args, times, party2loaders, global_train_dl, test_dl):
        super().__init__(args, times, party2loaders, global_train_dl, test_dl)
        # 诊断超参（写死，避免动 main.py 的 argparse）
        self.ccvr_samples_per_class = 500   # 每类采样的虚拟特征数（平衡）
        self.ccvr_head_epochs = 100         # 重训 head 的轮数
        self.ccvr_head_lr = 0.01
        self.ccvr_cov_eps = 1e-4            # 协方差正则，防奇异

    def train(self):
        # ===== 1. 先跑完整的 FedAvg =====
        super().train()

        print("\n" + "=" * 60)
        print("[CCVR 诊断] FedAvg 训练完成，开始病根诊断")
        print("=" * 60)

        # ===== 2. 诊断前：原 head 的整体 + per-class 精度 =====
        acc_before, pc_before = self._eval_per_class(self.global_model)
        print(f"[诊断·校准前] 整体 acc={acc_before:.4f}")
        print(f"[诊断·校准前] per_class={np.round(pc_before * 100, 1)}")

        # ===== 3. 收集每类特征统计量（μ_c, Σ_c）=====
        means, covs, feat_dim = self._collect_feature_stats()

        # ===== 4. 采样平衡虚拟特征，重训新 head =====
        new_head = self._retrain_head(means, covs, feat_dim)

        # ===== 5. 装回新 head，再测 =====
        calibrated = copy.deepcopy(self.global_model)
        calibrated.head = new_head
        acc_after, pc_after = self._eval_per_class(calibrated)
        print(f"[诊断·校准后] 整体 acc={acc_after:.4f}")
        print(f"[诊断·校准后] per_class={np.round(pc_after * 100, 1)}")

        # ===== 6. 结论 =====
        delta = (acc_after - acc_before) * 100
        print("\n" + "=" * 60)
        print(f"[诊断结论] 仅重训 head：{acc_before*100:.2f}% → {acc_after*100:.2f}%  (Δ={delta:+.2f} pt)")
        if delta >= 8.0:
            print(">>> 病根在 CLASSIFIER HEAD：feature 是好的，组件3(CCVR重建)是主力。")
        elif delta >= 3.0:
            print(">>> head 有偏但非全部：feature 也部分受损，组件1(去偏蒸馏)+组件3 都需要。")
        else:
            print(">>> head 重训提升很小：FEATURE 本身坏了，KD/重建都救不动，需转表征学习。")
        print("=" * 60)

    # ----------------------------------------------------------------
    def _extract_feat(self, x):
        if isinstance(x, list):
            x[0] = x[0].to(self.device)
            return self.global_model.base(x[0])
        return self.global_model.base(x.to(self.device))

    def _collect_feature_stats(self):
        """遍历所有 client 训练数据，按类累计特征，算 μ_c 和 Σ_c。"""
        self.global_model.eval()
        feats_by_class = {c: [] for c in range(self.num_classes)}
        feat_dim = None

        with torch.no_grad():
            for cid in range(self.num_clients):
                loader = self.party2loaders_train[cid]
                for x, y in loader:
                    feat = self._extract_feat(x).cpu()
                    if feat_dim is None:
                        feat_dim = feat.size(1)
                    y = y.long()
                    for c in range(self.num_classes):
                        m = (y == c)
                        if m.sum() > 0:
                            feats_by_class[c].append(feat[m])

        means, covs = {}, {}
        for c in range(self.num_classes):
            if len(feats_by_class[c]) == 0:
                continue
            f = torch.cat(feats_by_class[c], dim=0)            # [N_c, D]
            means[c] = f.mean(dim=0)
            if f.size(0) > 1:
                cov = torch.from_numpy(np.cov(f.numpy(), rowvar=False)).float()
            else:
                cov = torch.eye(feat_dim)
            covs[c] = cov + self.ccvr_cov_eps * torch.eye(feat_dim)
            print(f"  [stats] 类{c}: N={f.size(0)}")
        return means, covs, feat_dim

    def _retrain_head(self, means, covs, feat_dim):
        """从每类高斯采样平衡虚拟特征，训练一个全新 linear head。"""
        vfeats, vlabels = [], []
        M = self.ccvr_samples_per_class
        for c in range(self.num_classes):
            if c not in means:
                continue
            dist = MultivariateNormal(means[c], covariance_matrix=covs[c])
            s = dist.sample((M,))                              # [M, D]
            vfeats.append(s)
            vlabels.append(torch.full((M,), c, dtype=torch.long))
        vfeats = torch.cat(vfeats, dim=0).to(self.device)
        vlabels = torch.cat(vlabels, dim=0).to(self.device)

        new_head = nn.Linear(feat_dim, self.num_classes).to(self.device)
        opt = torch.optim.SGD(new_head.parameters(), lr=self.ccvr_head_lr, momentum=0.9)
        crit = nn.CrossEntropyLoss()

        idx = torch.randperm(vfeats.size(0))
        vfeats, vlabels = vfeats[idx], vlabels[idx]
        bs = 256
        new_head.train()
        for ep in range(self.ccvr_head_epochs):
            for i in range(0, vfeats.size(0), bs):
                xb = vfeats[i:i + bs]
                yb = vlabels[i:i + bs]
                opt.zero_grad()
                loss = crit(new_head(xb), yb)
                loss.backward()
                opt.step()
        return new_head

    def _eval_per_class(self, model):
        """返回 (整体acc, per_class_acc[num_classes])。"""
        model.eval()
        correct = np.zeros(self.num_classes)
        total = np.zeros(self.num_classes)
        with torch.no_grad():
            for x, y in self.party2loaders_test:
                x = x.to(self.device)
                y = y.long().to(self.device)
                pred = model(x).argmax(dim=1)
                for c in range(self.num_classes):
                    m = (y == c)
                    total[c] += m.sum().item()
                    correct[c] += ((pred == y) & m).sum().item()
        pc = correct / np.maximum(total, 1)
        overall = correct.sum() / np.maximum(total.sum(), 1)
        return overall, pc