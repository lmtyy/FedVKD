"""
flcore/servers/servervkd.py — FedVKD 服务端（v3: Data-Aware Selective KD）

核心改动：
  1. set_clients() 后调用 _init_distill_weights()，
     根据每个 client 的本地数据分布计算蒸馏权重
  2. wandb 上报字段适配 v3 client 的 metrics
  3. 新增 per-class accuracy 评估（用于论文公平性分析）
  4. 每 10 轮打印 DIAG 信息
"""
import time
import torch
import torch.nn as nn
import numpy as np
import copy

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
        """遍历每个 client 的 dataloader，统计类别分布，计算蒸馏权重。"""
        print("\n[Data-Aware] Computing distillation weights per client...")

        # 统计每个 client 每个类的样本数
        client_class_counts = np.zeros((self.num_clients, self.num_classes))
        for cid in range(self.num_clients):
            loader = party2loaders[cid]
            for _, targets in loader:
                for t in targets.numpy():
                    client_class_counts[cid][t] += 1

        # 计算全局平均每类样本数
        total_samples = client_class_counts.sum()
        global_avg_per_class = total_samples / self.num_classes / self.num_clients
        print(f"[Data-Aware] Total samples: {int(total_samples)}, "
              f"Global avg per class per client: {global_avg_per_class:.1f}")

        # 为每个 client 设置蒸馏权重
        for cid, client in enumerate(self.clients):
            client.set_distill_weights(
                local_class_counts=client_class_counts[cid],
                global_avg_per_class=global_avg_per_class
            )

        # 打印汇总统计
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
            test_acc, test_loss = self.compute_accuracy(self.global_model, self.party2loaders_test)

            self.rs_test_acc.append(test_acc)
            self.Budget.append(time.time() - s_t)

            # 每 10 轮打印详细诊断
            if round_idx % 10 == 0 or round_idx == self.global_rounds - 1:
                best_acc = max(self.rs_test_acc)
                # 计算 per-class accuracy
                per_class_acc = self._compute_per_class_accuracy(
                    self.global_model, self.party2loaders_test)
                per_class_str = " ".join([f"{a:.1f}" for a in (per_class_acc * 100)])
                std_acc = per_class_acc.std().item() * 100

                # 汇总 client 蒸馏信息
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
                # 普通轮次简洁输出
                avg_alpha = np.mean([c.last_round_metrics.get("alpha", 0)
                                     for c in self.selected_clients])
                avg_distill_cls = np.mean([c.last_round_metrics.get("num_distill_classes", 0)
                                           for c in self.selected_clients])
                print(f"Round {round_idx}/{self.global_rounds} | "
                      f"Loss: {test_loss:.4f} | Test: {test_acc*100:.2f}% | "
                      f"α={avg_alpha:.3f} distill_cls={avg_distill_cls:.1f} | "
                      f"Time: {self.Budget[-1]:.1f}s")

            self._log_wandb(round_idx, test_acc, test_loss)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # 最终输出
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

    # ================================================================
    #                  Per-Class Accuracy 评估
    # ================================================================
    @torch.no_grad()
    def _compute_per_class_accuracy(self, model, dataloader):
        """计算每个类的准确率，返回 numpy array [C]。"""
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

        # 避免除零
        total = np.maximum(total, 1)
        return correct / total

    # ================================================================
    #                       wandb 上报
    # ================================================================
    def _log_wandb(self, round_idx, test_acc, test_loss):
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

        # 上报 per-class accuracy
        per_class_acc = self._compute_per_class_accuracy(
            self.global_model, self.party2loaders_test)
        for c in range(self.num_classes):
            log_dict[f"per_class/class_{c}_acc"] = per_class_acc[c]
        log_dict["per_class/std"] = per_class_acc.std()
        log_dict["per_class/min"] = per_class_acc.min()
        log_dict["per_class/max"] = per_class_acc.max()

        # 上报 client 平均指标
        keys = ["total_loss", "ce_loss", "kd_loss", "alpha",
                "use_distill", "num_distill_classes",
                "num_full_distill_classes", "distill_weight_mean"]
        for k in keys:
            vals = [c.last_round_metrics[k] for c in self.selected_clients
                    if hasattr(c, 'last_round_metrics') and k in c.last_round_metrics]
            if vals:
                log_dict[f"client/{k}_avg"] = float(np.mean(vals))

        wandb.log(log_dict, step=round_idx)

    # ================================================================
    #                       评估
    # ================================================================
    def compute_accuracy(self, model, dataloader):
        was_training = False
        if model.training:
            model.eval()
            was_training = True

        correct, total = 0, 0
        criterion = nn.CrossEntropyLoss()
        loss_collector = []
        with torch.no_grad():
            for x, target in dataloader:
                x, target = x.to(self.device), target.to(dtype=torch.int64).to(self.device)
                out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
            avg_loss = sum(loss_collector) / max(len(loss_collector), 1)

        if was_training:
            model.train()
        return correct / float(total), avg_loss

    # ================================================================
    #                  client 管理 + 通信
    # ================================================================
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

    # ================================================================
    #                    auto_break 收敛检测
    # ================================================================
    def check_done(self, acc_lss, top_cnt=100):
        for acc_ls in acc_lss:
            if len(acc_ls) < top_cnt:
                return False
            recent = acc_ls[-top_cnt:]
            history = acc_ls[:-top_cnt] + [0.0]
            if max(recent) <= max(history):
                return True
        return False