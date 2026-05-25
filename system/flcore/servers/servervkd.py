import time
from flcore.clients.clientvkd import clientVKD
import torch.nn as nn
import torch
import numpy as np
import copy

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class FedVKD(object):
    """
    FedVKD 服务器端
    
    服务器端逻辑与 FedAvg 完全一致（标准 FedAvg 聚合），
    所有创新都在客户端。这意味着 FedVKD 不增加任何通信开销。
    """

    def __init__(self, args, times, party2loaders, global_train_dl, test_dl):
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.global_model = copy.deepcopy(args.model)
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

        # ★ WandB 开关
        self.use_wandb = getattr(args, 'use_wandb', False) and HAS_WANDB

        # 初始化客户端
        self.set_clients(clientVKD, party2loaders)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"WandB logging: {'ON' if self.use_wandb else 'OFF'}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        """FedVKD 主训练循环"""
        for round in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n-------------Round number: {round}-------------")
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            print(f"-------------{current_time}-------------")

            # 各客户端本地训练
            for client in self.selected_clients:
                client.train(self.party2loaders_train[client.id], round)

            # 收集并聚合
            self.receive_models()
            self.aggregate_parameters()

            # 评估
            print("\nEvaluate aggregated global model")
            test_acc, test_loss = self.compute_accuracy(
                self.global_model, self.party2loaders_test
            )
            print('>> Aggregated global model test accuracy : %f test loss: ' % test_acc, test_loss)
            self.rs_test_acc.append(test_acc)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            # ★★★ WandB 日志上报 ★★★
            if self.use_wandb:
                log_dict = {
                    "round": round + 1,
                    "global/test_acc": test_acc,
                    "global/test_loss": test_loss,
                    "global/time_per_round": self.Budget[-1],
                }

                # 收集客户端训练指标
                client_metrics = self._collect_client_metrics()
                log_dict.update({
                    "clients/avg_total_loss": client_metrics["avg_total_loss"],
                    "clients/avg_ce_loss": client_metrics["avg_ce_loss"],
                    "clients/avg_kd_loss": client_metrics["avg_kd_loss"],
                    "clients/avg_vulnerability_mean": client_metrics["avg_vuln_mean"],
                    "clients/avg_vulnerability_max": client_metrics["avg_vuln_max"],
                    "clients/num_vulnerable_classes": client_metrics["avg_num_vuln_classes"],
                })

                # 每类准确率
                per_class_acc = self._compute_per_class_accuracy()
                if per_class_acc is not None:
                    log_dict["global/worst_class_acc"] = min(per_class_acc)
                    log_dict["global/best_class_acc"] = max(per_class_acc)
                    log_dict["global/class_acc_std"] = float(np.std(per_class_acc))
                    for c, acc_c in enumerate(per_class_acc):
                        log_dict[f"per_class/class_{c}"] = acc_c

                wandb.log(log_dict, step=round + 1)
            # ★★★ WandB 日志上报结束 ★★★

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

    # ★★★ 新增方法：收集客户端指标 ★★★
    def _collect_client_metrics(self):
        """收集所有参与客户端的训练指标（用于 wandb）"""
        metrics = {
            "avg_total_loss": 0.0,
            "avg_ce_loss": 0.0,
            "avg_kd_loss": 0.0,
            "avg_vuln_mean": 0.0,
            "avg_vuln_max": 0.0,
            "avg_num_vuln_classes": 0.0,
        }
        n = len(self.selected_clients)
        if n == 0:
            return metrics

        for client in self.selected_clients:
            if hasattr(client, 'last_round_metrics'):
                m = client.last_round_metrics
                metrics["avg_total_loss"] += m.get("total_loss", 0.0)
                metrics["avg_ce_loss"] += m.get("ce_loss", 0.0)
                metrics["avg_kd_loss"] += m.get("kd_loss", 0.0)
                metrics["avg_vuln_mean"] += m.get("vuln_mean", 0.0)
                metrics["avg_vuln_max"] += m.get("vuln_max", 0.0)
                metrics["avg_num_vuln_classes"] += m.get("num_vuln_classes", 0.0)

        for k in metrics:
            metrics[k] /= n
        return metrics

    # ★★★ 新增方法：计算每类准确率 ★★★
    def _compute_per_class_accuracy(self):
        """计算全局模型在测试集上的每类准确率"""
        self.global_model.eval()
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes

        with torch.no_grad():
            for x, target in self.party2loaders_test:
                x = x.to(self.device)
                target = target.to(dtype=torch.int64).to(self.device)
                out = self.global_model(x)
                _, pred = torch.max(out, 1)

                for c in range(self.num_classes):
                    mask = (target == c)
                    class_total[c] += mask.sum().item()
                    class_correct[c] += (pred[mask] == target[mask]).sum().item()

        per_class_acc = []
        for c in range(self.num_classes):
            if class_total[c] > 0:
                per_class_acc.append(class_correct[c] / class_total[c])
            else:
                per_class_acc.append(0.0)

        self.global_model.train()  # ★ 恢复训练状态
        return per_class_acc

    def compute_accuracy(self, model, dataloader):
        """计算全局模型在测试集上的准确率"""
        was_training = False
        if model.training:
            model.eval()
            was_training = True

        correct, total = 0, 0
        criterion = nn.CrossEntropyLoss()
        loss_collector = []

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                x, target = x.to(self.device), target.to(dtype=torch.int64).to(self.device)
                out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

        avg_loss = sum(loss_collector) / len(loss_collector)

        if was_training:
            model.train()

        return correct / float(total), avg_loss

    def set_clients(self, clientObj, party2loaders):
        """初始化所有客户端"""
        for i in range(self.num_clients):
            dataload = party2loaders[i]
            client = clientObj(
                self.args,
                id=i,
                train_samples=len(dataload.dataset),
            )
            self.clients.append(client)

    def select_clients(self):
        """随机选择参与本轮训练的客户端"""
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients

        selected_clients = list(np.random.choice(
            self.clients, self.current_num_join_clients, replace=False
        ))
        return selected_clients

    def send_models(self):
        """将全局模型下发给选中的客户端"""
        assert (len(self.clients) > 0)
        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        """收集客户端训练后的模型"""
        assert (len(self.selected_clients) > 0)

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
        """FedAvg 加权聚合"""
        assert (len(self.uploaded_models) > 0)

        global_model_w = self.global_model.state_dict()
        temp = True

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            client_model_w = client_model.state_dict()
            if temp:
                for key in client_model_w:
                    global_model_w[key] = client_model_w[key] * w
                temp = False
            else:
                for key in client_model_w:
                    global_model_w[key] += client_model_w[key] * w

        self.global_model.load_state_dict(global_model_w)

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        """检查是否收敛（用于 auto_break）"""
        for acc_ls in acc_lss:
            if top_cnt is not None and top_cnt > 0:
                if len(acc_ls) >= top_cnt:
                    recent = acc_ls[-top_cnt:]
                    if max(recent) - min(recent) < 1e-4:
                        return True
        return False
