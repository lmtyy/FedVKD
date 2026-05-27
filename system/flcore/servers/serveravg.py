"""
flcore/servers/serveravg.py — FedAvg 服务端（修复版 v2）

相对仓库现版的改动：
1. ★ Budget[1:] 平均时长加保护，避免 auto_break 后 ZeroDivisionError
2. ★ rs_test_acc 为空时不打印 max(...)，避免 ValueError
3. wandb 上报增加 best_acc
"""
import time
import torch
import torch.nn as nn
import numpy as np
import copy

from flcore.clients.clientavg import clientAVG


class FedAvg(object):
    def __init__(self, args, times, party2loaders, global_train_dl, test_dl):
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
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

        self.set_clients(clientAVG, party2loaders)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.use_wandb = getattr(args, 'use_wandb', False)

    def train(self):
        for round_idx in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            #print(f"\n-------------Round number: {round_idx}-------------")
            #current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            #print(f"-------------{current_time}-------------")

            # clientAVG.train 只收一个参数（trainloader）
            for client in self.selected_clients:
                client.train(self.party2loaders_train[client.id])

            self.receive_models()
            self.aggregate_parameters()

            #print("\nEvaluate aggregated global model")
            test_acc, test_loss = self.compute_accuracy(self.global_model, self.party2loaders_test)
            #print('>> Aggregated global model test accuracy : %f test loss: %f' % (test_acc, test_loss))

            self.rs_test_acc.append(test_acc)
            self.Budget.append(time.time() - s_t)
            is_milestone = (round_idx % 10 == 0) or (round_idx == self.global_rounds - 1)
            loss = test_loss  # 或者从 client 收集 train loss，这里偷懒用 test_loss
            if is_milestone:
                best = max(self.rs_test_acc) if self.rs_test_acc else 0.0
                print(f"Round {round_idx:3d}/{self.global_rounds} | "
                    f"Loss: {loss:.4f} | Test Acc: {test_acc*100:.2f}% | "
                    f"Best: {best*100:.2f}% | Time: {self.Budget[-1]:.1f}s")
            else:
                print(f"Round {round_idx:3d}/{self.global_rounds} | "
                    f"Loss: {loss:.4f} | Test: {test_acc*100:.2f}% | "
                    f"Time: {self.Budget[-1]:.1f}s")

            self._log_wandb(round_idx, test_acc, test_loss)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # ★ 防御式打印
        if self.rs_test_acc:
            print("\nBest accuracy: %.4f" % max(self.rs_test_acc))
        else:
            print("\nNo accuracy recorded.")
        print("\nAverage time cost per round.")
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        elif self.Budget:
            print(self.Budget[0])
        else:
            print(0.0)

    def _log_wandb(self, round_idx, test_acc, test_loss):
        if not self.use_wandb:
            return
        try:
            import wandb
        except ImportError:
            return
        wandb.log({
            "server/test_acc": test_acc,
            "server/test_loss": test_loss,
            "server/best_acc": max(self.rs_test_acc) if self.rs_test_acc else test_acc,
            "server/round": round_idx,
        }, step=round_idx)

    def compute_accuracy(self, model, dataloader):
        was_training = model.training
        model.eval()
        # ★ 修复 BN 聚合污染：评估时让 BN 用当前 batch 统计而不是被 non-iid 污染的 running_stats
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()

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
