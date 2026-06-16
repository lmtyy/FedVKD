"""
flcore/servers/servervkd.py — FedVKD 服务端（Selective KD + optional HeadCal）
"""
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn

from flcore.clients.clientvkd import clientVKD


class FedVKD(object):
    def __init__(self, args, times, party2loaders, global_train_dl, test_dl):
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.global_model = copy.deepcopy(args.model)
        self.global_train_dl = global_train_dl

        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = min(self.num_clients, max(1, int(self.num_clients * self.join_ratio)))
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_train_loss = []
        self.Budget = []

        self.best_acc = -1.0
        self.best_round = -1
        self.best_model_state = None
        self.best_raw_model_state = None
        self.save_best = getattr(args, "save_best", True)
        self.checkpoint_dir = getattr(args, "checkpoint_dir", "./checkpoints")

        self.times = times
        self.party2loaders_train = party2loaders
        self.party2loaders_test = test_dl

        self.use_wandb = getattr(args, "use_wandb", False)
        self.use_headcal = getattr(args, "use_headcal", False)
        self.use_gdc = getattr(args, "use_gdc", False)
        self.prototype_bank = None
        self.prototype_valid = None
        self.client_class_counts = None

        self.set_clients(clientVKD, party2loaders)
        self._init_distill_weights(party2loaders)
        if self.use_gdc:
            self._build_prototype_bank(self.global_model, self.global_train_dl)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"HeadCal enabled: {self.use_headcal}")
        print(f"FedGDC-Lite enabled: {self.use_gdc}")
        print("Finished creating FedVKD server and clients.")

    def _init_distill_weights(self, party2loaders):
        print("\n[Data-Aware] Computing distillation weights per client...")
        client_class_counts = np.zeros((self.num_clients, self.num_classes))

        for cid in range(self.num_clients):
            loader = party2loaders[cid]
            for _, targets in loader:
                targets = targets.detach().cpu().numpy()
                for t in targets:
                    client_class_counts[cid][int(t)] += 1

        total_samples = client_class_counts.sum()
        global_avg_per_class = total_samples / max(self.num_classes * self.num_clients, 1)
        print(f"[Data-Aware] Total samples: {int(total_samples)}")
        print(f"[Data-Aware] Global avg per class per client: {global_avg_per_class:.1f}")

        for cid, client in enumerate(self.clients):
            client.set_distill_weights(
                local_class_counts=client_class_counts[cid],
                global_avg_per_class=global_avg_per_class,
            )
        self.client_class_counts = client_class_counts

        all_weights = torch.stack([c.distill_weights for c in self.clients])
        avg_num_distill = (all_weights > 0).float().sum(dim=1).mean().item()
        avg_num_full = (all_weights >= 0.99).float().sum(dim=1).mean().item()
        print(
            f"[Data-Aware] Avg distill classes per client: {avg_num_distill:.1f} "
            f"(full: {avg_num_full:.1f}) / {self.num_classes}"
        )

    @torch.no_grad()
    def _build_prototype_bank(self, model, dataloader):
        model = model.to(self.device)
        was_training = model.training
        model.eval()

        proto_sums = None
        proto_counts = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device)

        for x, target in dataloader:
            if isinstance(x, list):
                x = x[0]
            x = x.to(self.device)
            target = target.to(torch.int64).to(self.device)
            feature = model.base(x)
            if feature.dim() > 2:
                feature = torch.nn.functional.adaptive_avg_pool2d(feature, 1).flatten(1)
            feature = feature.detach()

            if proto_sums is None:
                proto_sums = torch.zeros(
                    self.num_classes,
                    feature.size(1),
                    dtype=feature.dtype,
                    device=self.device,
                )

            for c in range(self.num_classes):
                mask = (target == c)
                if mask.any():
                    proto_sums[c] += feature[mask].sum(dim=0)
                    proto_counts[c] += mask.sum().float()

        if proto_sums is None:
            self.prototype_bank = None
            self.prototype_valid = None
            return

        proto_valid = proto_counts > 0
        proto_bank = torch.zeros_like(proto_sums)
        proto_bank[proto_valid] = proto_sums[proto_valid] / proto_counts[proto_valid].unsqueeze(1)
        self.prototype_bank = proto_bank.detach().cpu()
        self.prototype_valid = proto_valid.detach().cpu()

        if was_training:
            model.train()

        print(
            f"[FedGDC-Lite] prototype_bank valid="
            f"{int(proto_valid.sum().item())}/{self.num_classes}"
        )

    def train(self):
        for round_idx in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n-------------Round number: {round_idx}-------------")
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            print(f"-------------{current_time}-------------")

            for client in self.selected_clients:
                client.train(self.party2loaders_train[client.id], round_idx)

            self.receive_models()
            self.aggregate_parameters()
            if self.use_gdc:
                self._build_prototype_bank(self.global_model, self.global_train_dl)

            headcal_metrics = {}
            if self._should_run_headcal(round_idx):
                headcal_metrics = self._run_headcal(round_idx)

            eval_model = copy.deepcopy(self.global_model)
            self.recalibrate_bn(eval_model, self.global_train_dl)
            test_acc, test_loss = self.compute_accuracy(eval_model, self.party2loaders_test)
            per_class_acc = self._compute_per_class_accuracy(eval_model, self.party2loaders_test)

            self.rs_test_acc.append(test_acc)
            self._update_best_checkpoint(round_idx, test_acc, eval_model)
            self.Budget.append(time.time() - s_t)

            self._print_round_log(round_idx, test_acc, test_loss, per_class_acc, headcal_metrics)
            self._log_wandb(round_idx, test_acc, test_loss, per_class_acc, headcal_metrics)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        if self.rs_test_acc:
            print(f"\nBest accuracy: {self.best_acc:.4f} at round {self.best_round}")
        else:
            print("\nNo accuracy recorded.")
        print()
        print("Average time cost per round.")
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        elif self.Budget:
            print(self.Budget[0])
        else:
            print(0.0)

    def _should_run_headcal(self, round_idx):
        if not getattr(self.args, "use_headcal", False):
            return False

        start_round = getattr(self.args, "headcal_start_round", -1)
        if start_round < 0:
            start_round = getattr(self.args, "warmup_rounds", 0)

        if round_idx < start_round:
            return False
        interval = max(getattr(self.args, "headcal_interval", 5), 1)
        return (round_idx - start_round) % interval == 0

    def _run_headcal(self, round_idx):
        before_model = copy.deepcopy(self.global_model).to(self.device)
        self.recalibrate_bn(before_model, self.global_train_dl)
        before_acc, before_loss = self.compute_accuracy(before_model, self.party2loaders_test)

        calib_model = copy.deepcopy(self.global_model).to(self.device)
        self.recalibrate_bn(calib_model, self.global_train_dl)
        calib_model.eval()

        for param in calib_model.base.parameters():
            param.requires_grad = False

        feats_by_class = {c: [] for c in range(self.num_classes)}
        feature_cap_per_class = max(getattr(self.args, "headcal_samples_per_class", 256) * 2, 1)
        feature_counts = np.zeros(self.num_classes, dtype=np.int64)
        with torch.no_grad():
            for x, y in self.global_train_dl:
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                y = y.long().to(self.device)
                feat = calib_model.base(x).detach().cpu()
                if feat.dim() > 2:
                    feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
                y_cpu = y.detach().cpu()
                for c in range(self.num_classes):
                    if feature_counts[c] >= feature_cap_per_class:
                        continue
                    mask = (y_cpu == c)
                    if mask.sum() > 0:
                        selected_feat = feat[mask]
                        remaining = feature_cap_per_class - feature_counts[c]
                        feats_by_class[c].append(selected_feat[:remaining])
                        feature_counts[c] += min(selected_feat.size(0), remaining)

        balanced_feats = []
        balanced_labels = []
        samples_per_class = getattr(self.args, "headcal_samples_per_class", 256)
        num_headcal_classes = 0

        for c in range(self.num_classes):
            if len(feats_by_class[c]) == 0:
                continue
            class_feats = torch.cat(feats_by_class[c], dim=0)
            n = class_feats.size(0)
            if n >= samples_per_class:
                idx = torch.randperm(n)[:samples_per_class]
            else:
                idx = torch.randint(0, n, (samples_per_class,))
            balanced_feats.append(class_feats[idx])
            balanced_labels.append(torch.full((samples_per_class,), c, dtype=torch.long))
            num_headcal_classes += 1

        if not balanced_feats:
            return {
                "headcal_ran": 0.0,
                "headcal_loss": 0.0,
                "headcal_classes": 0.0,
                "headcal_before_acc": before_acc,
                "headcal_after_acc": before_acc,
                "headcal_delta": 0.0,
                "headcal_before_loss": before_loss,
                "headcal_after_loss": before_loss,
            }

        balanced_feats = torch.cat(balanced_feats, dim=0).to(self.device)
        balanced_labels = torch.cat(balanced_labels, dim=0).to(self.device)

        new_head = copy.deepcopy(calib_model.head).to(self.device)
        new_head.train()
        optimizer = torch.optim.SGD(
            new_head.parameters(),
            lr=getattr(self.args, "headcal_lr", 0.01),
            momentum=0.9,
            weight_decay=getattr(self.args, "headcal_weight_decay", 0.0),
        )
        criterion = nn.CrossEntropyLoss()
        batch_size = max(getattr(self.args, "headcal_batch_size", 256), 1)
        losses = []

        for _ in range(getattr(self.args, "headcal_epochs", 5)):
            perm = torch.randperm(balanced_feats.size(0), device=self.device)
            for start in range(0, balanced_feats.size(0), batch_size):
                idx = perm[start:start + batch_size]
                xb = balanced_feats[idx]
                yb = balanced_labels[idx]
                optimizer.zero_grad()
                logits = new_head(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        self.global_model.head.load_state_dict(new_head.state_dict())
        avg_loss = float(np.mean(losses)) if losses else 0.0

        after_model = copy.deepcopy(self.global_model).to(self.device)
        self.recalibrate_bn(after_model, self.global_train_dl)
        after_acc, after_loss = self.compute_accuracy(after_model, self.party2loaders_test)
        delta = after_acc - before_acc

        print(
            f"[HeadCal] round={round_idx} ran=1 "
            f"before={before_acc*100:.2f}% after={after_acc*100:.2f}% "
            f"delta={delta*100:+.2f}pt loss={avg_loss:.4f} "
            f"classes={num_headcal_classes} samples={balanced_feats.size(0)}"
        )
        return {
            "headcal_ran": 1.0,
            "headcal_loss": avg_loss,
            "headcal_classes": float(num_headcal_classes),
            "headcal_before_acc": before_acc,
            "headcal_after_acc": after_acc,
            "headcal_delta": delta,
            "headcal_before_loss": before_loss,
            "headcal_after_loss": after_loss,
        }

    def _forward_model(self, model, x):
        if hasattr(model, "base") and hasattr(model, "head"):
            feature = model.base(x)
            if feature.dim() > 2:
                feature = torch.nn.functional.adaptive_avg_pool2d(feature, 1).flatten(1)
            return model.head(feature)
        return model(x)

    @torch.no_grad()
    def recalibrate_bn(self, model, loader, num_batches=50):
        model.train()
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.reset_running_stats()

        seen = 0
        for x, _ in loader:
            if seen >= num_batches:
                break
            if isinstance(x, list):
                x = x[0]
            x = x.to(self.device)
            if x.size(0) <= 1:
                continue
            self._forward_model(model, x)
            seen += 1
        model.eval()

    def compute_accuracy(self, model, dataloader):
        was_training = model.training
        model.eval()
        correct, total = 0, 0
        criterion = nn.CrossEntropyLoss()
        losses = []

        with torch.no_grad():
            for x, target in dataloader:
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                target = target.to(torch.int64).to(self.device)
                out = self._forward_model(model, x)
                loss = criterion(out, target)
                losses.append(loss.item())
                pred = out.argmax(dim=1)
                total += target.size(0)
                correct += (pred == target).sum().item()

        if was_training:
            model.train()
        return correct / max(float(total), 1.0), sum(losses) / max(len(losses), 1)

    @torch.no_grad()
    def _compute_per_class_accuracy(self, model, dataloader):
        was_training = model.training
        model.eval()
        correct = np.zeros(self.num_classes)
        total = np.zeros(self.num_classes)

        for x, target in dataloader:
            if isinstance(x, list):
                x = x[0]
            x = x.to(self.device)
            target = target.to(torch.int64).to(self.device)
            out = self._forward_model(model, x)
            pred = out.argmax(dim=1)
            for c in range(self.num_classes):
                mask = (target == c)
                total[c] += mask.sum().item()
                correct[c] += ((pred == c) & mask).sum().item()

        if was_training:
            model.train()
        return correct / np.maximum(total, 1)

    def _metric_avg(self, key):
        vals = [
            client.last_round_metrics.get(key, 0.0)
            for client in self.selected_clients
            if hasattr(client, "last_round_metrics")
        ]
        return float(np.mean(vals)) if vals else 0.0

    def _missing_class_accuracy(self, per_class_acc):
        if self.client_class_counts is None:
            return 0.0

        client_missing_acc = []
        for cid in range(self.num_clients):
            missing = np.where(self.client_class_counts[cid] <= 0)[0]
            if missing.size == 0:
                continue
            client_missing_acc.append(float(np.mean(per_class_acc[missing])))
        return float(np.mean(client_missing_acc)) if client_missing_acc else 0.0

    def _print_round_log(self, round_idx, test_acc, test_loss, per_class_acc, headcal_metrics):
        best_acc = max(self.rs_test_acc) if self.rs_test_acc else test_acc
        avg_alpha = self._metric_avg("alpha")
        avg_distill_cls = self._metric_avg("num_distill_classes")
        avg_effective_cls = self._metric_avg("num_effective_distill_classes")
        avg_vuln_cls = self._metric_avg("num_vulnerable_classes")
        headcal_ran = headcal_metrics.get("headcal_ran", 0.0)
        headcal_delta = headcal_metrics.get("headcal_delta", 0.0)
        macro_acc = float(np.mean(per_class_acc))
        worst_acc = float(np.min(per_class_acc))
        missing_acc = self._missing_class_accuracy(per_class_acc)
        gdc_loss = self._metric_avg("gdc_loss")
        gdc_classes = self._metric_avg("gdc_classes")
        gdc_debt = self._metric_avg("gdc_debt")

        print(
            f"Round {round_idx}/{self.global_rounds} | Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc*100:.2f}% | Best: {best_acc*100:.2f}% | "
            f"Macro: {macro_acc*100:.2f}% | Worst: {worst_acc*100:.2f}% | "
            f"Missing: {missing_acc*100:.2f}% | "
            f"alpha={avg_alpha:.3f} distill_cls={avg_distill_cls:.1f} | "
            f"effective_cls={avg_effective_cls:.1f} vuln_cls={avg_vuln_cls:.1f} | "
            f"GDC_loss={gdc_loss:.4f} GDC_cls={gdc_classes:.1f} debt={gdc_debt:.3f} | "
            f"HeadCal={headcal_ran:.0f} Δ={headcal_delta*100:+.2f}pt | "
            f"Time: {self.Budget[-1]:.1f}s"
        )

        is_milestone = (round_idx % 10 == 0) or (round_idx == self.global_rounds - 1)
        if is_milestone:
            per_class_str = " ".join([f"{acc * 100:.1f}" for acc in per_class_acc])
            print(
                f"    [DIAG] per_class_acc=[{per_class_str}] "
                f"macro={macro_acc*100:.2f} missing={missing_acc*100:.2f} "
                f"std={per_class_acc.std()*100:.2f} min={per_class_acc.min()*100:.2f} "
                f"max={per_class_acc.max()*100:.2f}"
            )
            print(
                f"    [DIAG] ce_loss={self._metric_avg('ce_loss'):.4f} "
                f"kd_loss={self._metric_avg('kd_loss'):.4f} "
                f"logit_kd={self._metric_avg('logit_kd_loss'):.4f} "
                f"feat_kd={self._metric_avg('feat_kd_loss'):.4f}"
            )
            print(
                f"    [DIAG] vulnerability_mean={self._metric_avg('vulnerability_mean'):.4f} "
                f"vulnerability_max={self._metric_avg('vulnerability_max'):.4f}"
            )
            print(
                f"    [FedGDC-Lite] loss={gdc_loss:.4f} "
                f"classes={gdc_classes:.1f} debt={gdc_debt:.3f} "
                f"missing_acc={missing_acc*100:.2f}%"
            )
            print(
                f"    [HeadCal] ran={headcal_metrics.get('headcal_ran', 0.0):.0f} "
                f"before={headcal_metrics.get('headcal_before_acc', 0.0)*100:.2f}% "
                f"after={headcal_metrics.get('headcal_after_acc', 0.0)*100:.2f}% "
                f"delta={headcal_metrics.get('headcal_delta', 0.0)*100:+.2f}pt "
                f"loss={headcal_metrics.get('headcal_loss', 0.0):.4f} "
                f"classes={headcal_metrics.get('headcal_classes', 0.0):.0f}"
            )

    def _log_wandb(self, round_idx, test_acc, test_loss, per_class_acc, headcal_metrics):
        if not self.use_wandb:
            return
        try:
            import wandb
        except ImportError:
            return

        log_dict = {
            "server/test_acc": test_acc,
            "server/test_loss": test_loss,
            "server/best_acc": max(self.rs_test_acc) if self.rs_test_acc else test_acc,
            "server/macro_acc": float(np.mean(per_class_acc)),
            "server/worst_class_acc": float(np.min(per_class_acc)),
            "server/missing_class_acc": self._missing_class_accuracy(per_class_acc),
            "server/round": round_idx,
            "headcal/ran": headcal_metrics.get("headcal_ran", 0.0),
            "headcal/loss": headcal_metrics.get("headcal_loss", 0.0),
            "headcal/classes": headcal_metrics.get("headcal_classes", 0.0),
            "headcal/before_acc": headcal_metrics.get("headcal_before_acc", 0.0),
            "headcal/after_acc": headcal_metrics.get("headcal_after_acc", 0.0),
            "headcal/delta": headcal_metrics.get("headcal_delta", 0.0),
            "headcal/before_loss": headcal_metrics.get("headcal_before_loss", 0.0),
            "headcal/after_loss": headcal_metrics.get("headcal_after_loss", 0.0),
            "per_class/std": per_class_acc.std(),
            "per_class/min": per_class_acc.min(),
            "per_class/max": per_class_acc.max(),
        }
        for c in range(self.num_classes):
            log_dict[f"per_class/class_{c}_acc"] = per_class_acc[c]

        keys = [
            "total_loss",
            "ce_loss",
            "kd_loss",
            "logit_kd_loss",
            "feat_kd_loss",
            "alpha",
            "use_distill",
            "num_distill_classes",
            "num_effective_distill_classes",
            "num_vulnerable_classes",
            "distill_weight_mean",
            "effective_weight_mean",
            "effective_weight_max",
            "vulnerability_mean",
            "vulnerability_max",
            "gdc_loss",
            "gdc_classes",
            "gdc_debt",
        ]
        for key in keys:
            log_dict[f"client/{key}_avg"] = self._metric_avg(key)

        wandb.log(log_dict, step=round_idx)

    def _update_best_checkpoint(self, round_idx, test_acc, eval_model):
        if test_acc <= self.best_acc:
            return

        self.best_acc = test_acc
        self.best_round = round_idx
        self.best_model_state = copy.deepcopy(eval_model.state_dict())
        self.best_raw_model_state = copy.deepcopy(self.global_model.state_dict())

        if not self.save_best:
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        model_name = getattr(self.args, "model_name", None)
        if model_name is None:
            model_name = getattr(self.args, "model", "model")
            if not isinstance(model_name, str):
                model_name = "model"

        ckpt_name = (
            f"{self.algorithm}_{self.dataset}_"
            f"{model_name}_"
            f"alpha{getattr(self.args, 'alpha', 'na')}_"
            f"round{round_idx}_best.pt"
        )
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)

        safe_args = {
            key: value for key, value in vars(self.args).items()
            if isinstance(value, (int, float, str, bool, type(None)))
        }

        torch.save({
            "round": self.best_round,
            "best_acc": self.best_acc,
            "model_state": self.best_model_state,
            "raw_model_state": self.best_raw_model_state,
            "state_note": (
                "model_state is BN-recalibrated eval_model used for reported best_acc; "
                "raw_model_state is the non-recalibrated aggregated global_model."
            ),
            "args": safe_args,
        }, ckpt_path)

        print(
            f"[Checkpoint] New best {self.best_acc*100:.2f}% "
            f"at round {round_idx}, saved BN-recalibrated eval model to {ckpt_path}"
        )

    def set_clients(self, clientObj, party2loaders):
        for i in range(self.num_clients):
            dataload = party2loaders[i]
            client = clientObj(self.args, id=i, train_samples=len(dataload.dataset))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        return list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

    def send_models(self):
        assert len(self.clients) > 0
        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            if self.use_gdc:
                client.set_prototype_bank(self.prototype_bank, self.prototype_valid)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert len(self.selected_clients) > 0
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
        if tot_samples <= 0:
            raise ValueError("Total uploaded client samples must be positive.")
        for i, weight in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = weight / tot_samples

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0
        first_client_state = self.uploaded_models[0].state_dict()
        global_model_w = self.global_model.state_dict()
        for key in global_model_w:
            first_tensor = first_client_state[key]
            if not first_tensor.is_floating_point():
                global_model_w[key] = first_tensor.clone()
                continue

            aggregated = torch.zeros_like(first_tensor)
            for weight, client_model in zip(self.uploaded_weights, self.uploaded_models):
                aggregated += client_model.state_dict()[key] * weight
            global_model_w[key] = aggregated
        self.global_model.load_state_dict(global_model_w)

    def check_done(self, acc_lss, top_cnt=100):
        for acc_ls in acc_lss:
            if len(acc_ls) < top_cnt:
                return False
            recent = acc_ls[-top_cnt:]
            history = acc_ls[:-top_cnt] + [0.0]
            if max(recent) <= max(history):
                return True
        return False
