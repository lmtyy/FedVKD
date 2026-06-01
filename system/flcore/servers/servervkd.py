"""
flcore/servers/servervkd.py — FedVKD 服务端（v3 + P1 方案B BN重校准）

P1 改动：
1. __init__ 保存 self.global_train_dl
2. 新增 recalibrate_bn；评估在 global_model 的副本上进行
3. per-class accuracy / wandb 全部基于重校准后的 eval_model，口径统一
4. compute_accuracy 统一为纯 eval
"""
import time
import copy
import torch
import torch.nn as nn
import numpy as np

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
        self.teacher_model = None

        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
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
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.party2loaders_train = party2loaders
        self.party2loaders_test = test_dl
        self.global_train_dl = global_train_dl       # ★ P1: BN 重校准用全局训练集

        self.set_clients(clientVKD, party2loaders)

        # ★ Data-Aware: 初始化每个 client 的蒸馏权重
        self._init_distill_weights(party2loaders)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedVKD server and clients.")

        self.Budget = []
        self.use_wandb = getattr(args, 'use_wandb', False)

    # ================================================================
    #          Data-Aware: 计算每个 client 的蒸馏权重
    # ================================================================
    def _init_distill_weights(self, party2loaders):
        print("\n[Data-Aware] Computing distillation weights per client...")
        client_class_counts = np.zeros((self.num_clients, self.num_classes))
        for cid in range(self.num_clients):
            loader = party2loaders[cid]
            for _, targets in loader:
                for t in targets.numpy():
                    client_class_counts[cid][t] += 1

        total_samples = client_class_counts.sum()
        global_avg_per_class = total_samples / self.num_classes / self.num_clients
        print(f"[Data-Aware] Total samples: {int(total_samples)}, "
              f"Global avg per class per client: {global_avg_per_class:.1f}")

        for cid, client in enumerate(self.clients):
            client.set_distill_weights(
                local_class_counts=client_class_counts[cid],
                global_avg_per_class=global_avg_per_class
            )

        all_weights = torch.stack([c.distill_weights for c in self.clients])
        avg_num_distill = (all_weights > 0).float().sum(dim=1).mean().item()
        avg_num_full = (all_weights >= 0.99).float().sum(dim=1).mean().item()
        print(f"[Data-Aware] Avg distill classes per client: {avg_num_distill:.1f} "
              f"(full: {avg_num_full:.1f}) / {self.num_classes}")

    # ================================================================
    #                       主训练循环
    # ================================================================
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

            print("\nEvaluate aggregated global model")
            # ★ P1: 副本重校准 BN，后续所有评估都用 eval_model
            eval_model = copy.deepcopy(self.global_model)
            self.recalibrate_bn(eval_model, self.global_train_dl)
            test_acc, test_loss = self.compute_accuracy(eval_model, self.party2loaders_test)

            self.rs_test_acc.append(test_acc)
            self.Budget.append(time.time() - s_t)

            if round_idx % 10 == 0 or round_idx == self.global_rounds - 1:
                best_acc = max(self.rs_test_acc)
                per_class_acc = self._compute_per_class_accuracy(eval_model, self.party2loaders_test)
                per_class_str = " ".join([f"{a:.1f}" for a in (per_class_acc * 100)])
                std_acc = per_class_acc.std().item() * 100

                avg_kd = np.mean([c.last_round_metrics.get("kd_loss", 0)
                                  for c in self.selected_clients])
                avg_ce = np.mean([c.last_round_metrics.get("ce_loss", 0)
                                  for c in self.selected_clients])
                avg_distill_cls = np.mean([c.last_round_metrics.get("num_distill_classes", 0)
                                           for c in self.selected_clients])

                print(f"Round {round_idx}/{self.global_rounds} | "
                      f"Test Acc: {test_acc*100:.2f}% | Best: {best_acc*100:.2f}% | "
                      f"distill_cls={avg_distill_cls:.1f} | Time: {self.Budget[-1]:.1f}s")
                print(f"    [DIAG] per_class_acc=[{per_class_str}] std={std_acc:.2f} | "
                      f"kd_loss={avg_kd:.4f} ce_loss={avg_ce:.4f}")
            else:
                avg_alpha = np.mean([c.last_round_metrics.get("alpha", 0)
                                     for c in self.selected_clients])
                avg_distill_cls = np.mean([c.last_round_metrics.get("num_distill_classes", 0)
                                           for c in self.selected_clients])
                print(f"Round {round_idx}/{self.global_rounds} | "
                      f"Loss: {test_loss:.4f} | Test: {test_acc*100:.2f}% | "
                      f"α={avg_alpha:.3f} distill_cls={avg_distill_cls:.1f} | "
                      f"Time: {self.Budget[-1]:.1f}s")

            self._log_wandb(round_idx, test_acc, test_loss, eval_model)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        if self.rs_test_acc:
            print(f"\nBest accuracy: {max(self.rs_test_acc):.4f}")
        else:
            print("\nNo accuracy recorded.")
        print("\nAverage time cost per round.")
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        elif self.Budget:
            print(self.Budget[0])
        else:
            print(0.0)

    # ★ P1 核心：BN 重校准
    @torch.no_grad()
    def recalibrate_bn(self, model, loader, num_batches=50):
        model.train()
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.reset_running_stats()
        seen = 0
        for x, _ in loader:
            if seen >= num_batches:
                break
            if isinstance(x, list):
                x[0] = x[0].to(self.device)
                bs = x[0].size(0)
            else:
                x = x.to(self.device)
                bs = x.size(0)
            if bs <= 1:
                continue
            model(x)
            seen += 1
        model.eval()

    @torch.no_grad()
    def _compute_per_class_accuracy(self, model, dataloader):
        was_training = model.training
        model.eval()
        correct = np.zeros(self.num_classes)
        total = np.zeros(self.num_classes)
        for x, target in dataloader:
            x = x.to(self.device)
            target = target.to(dtype=torch.int64).to(self.device)
            out = model(x)
            _, pred = torch.max(out, 1)
            for c in range(self.num_classes):
                mask = (target == c)
                total[c] += mask.sum().item()
                correct[c] += ((pred == c) & mask).sum().item()
        if was_training:
            model.train()
        total = np.maximum(total, 1)
        return correct / total

    def _log_wandb(self, round_idx, test_acc, test_loss, eval_model):
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
            "server/round": round_idx,
        }

        per_class_acc = self._compute_per_class_accuracy(eval_model, self.party2loaders_test)
        for c in range(self.num_classes):
            log_dict[f"per_class/class_{c}_acc"] = per_class_acc[c]
        log_dict["per_class/std"] = per_class_acc.std()
        log_dict["per_class/min"] = per_class_acc.min()
        log_dict["per_class/max"] = per_class_acc.max()

        keys = ["total_loss", "ce_loss", "kd_loss", "alpha",
                "use_distill", "num_distill_classes",
                "num_full_distill_classes", "distill_weight_mean"]
        for k in keys:
            vals = [c.last_round_metrics[k] for c in self.selected_clients
                    if hasattr(c, 'last_round_metrics') and k in c.last_round_metrics]
            if vals:
                log_dict[f"client/{k}_avg"] = float(np.mean(vals))

        wandb.log(log_dict, step=round_idx)

    # ★ P1：统一的纯 eval 评估（返回 2 个值）
    def compute_accuracy(self, model, dataloader):
        was_training = model.training
        model.eval()
        correct, total = 0, 0
        criterion = nn.CrossEntropyLoss()
        loss_collector = []
        with torch.no_grad():
            for x, target in dataloader:
                x, target = x.to(self.device), target.to(torch.int64).to(self.device)
                out = model(x)
                loss_collector.append(criterion(out, target).item())
                _, pred = torch.max(out, 1)
                total += target.size(0)
                correct += (pred == target).sum().item()
        if was_training:
            model.train()
        return correct / float(total), sum(loss_collector) / max(len(loss_collector), 1)

    def set_clients(self, clientObj, party2loaders):
        for i in range(self.num_clients):
            dataload = party2loaders[i]
            client = clientObj(self.args, id=i, train_samples=len(dataload.dataset))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        return selected

    def send_models(self):
        assert len(self.clients) > 0
        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
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
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0
        global_model_w = self.global_model.state_dict()
        first = True
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            client_model_w = client_model.state_dict()
            if first:
                for key in client_model_w:
                    global_model_w[key] = client_model_w[key] * w
                first = False
            else:
                for key in client_model_w:
                    global_model_w[key] += client_model_w[key] * w
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