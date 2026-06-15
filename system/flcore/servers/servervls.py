"""
flcore/servers/servervls.py — FedVLS 服务端（P1 方案B：BN 重校准评估口径）

P1 改动：删除旧 BN hack；新增 recalibrate_bn；评估在 global_model 副本上进行；
compute_accuracy 统一为纯 eval；__init__ 保存 self.global_train_dl。
"""
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from scipy import special
from scipy.special import kl_div

from flcore.clients.clientvls import clientVLS

class FedVLS(object):
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
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.party2loaders_train = party2loaders
        self.party2loaders_test = test_dl
        self.global_train_dl = global_train_dl       # ★ P1

        self.set_clients(clientVLS, party2loaders)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        self.use_wandb = getattr(args, 'use_wandb', False)

    def train(self):
        for round_idx in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            for client in self.selected_clients:
                client.train(self.party2loaders_train[client.id], round_idx)

            self.receive_models()
            self.aggregate_parameters()

            # ★ P1: 副本重校准 BN 再评估
            eval_model = copy.deepcopy(self.global_model)
            self.recalibrate_bn(eval_model, self.global_train_dl)
            test_acc, test_loss = self.compute_accuracy(eval_model, self.party2loaders_test)

            self.rs_test_acc.append(test_acc)
            self.Budget.append(time.time() - s_t)
            is_milestone = (round_idx % 10 == 0) or (round_idx == self.global_rounds - 1)
            if is_milestone:
                best = max(self.rs_test_acc) if self.rs_test_acc else 0.0
                print(f"Round {round_idx:3d}/{self.global_rounds} | "
                      f"Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}% | "
                      f"Best: {best*100:.2f}% | Time: {self.Budget[-1]:.1f}s")
            else:
                print(f"Round {round_idx:3d}/{self.global_rounds} | "
                      f"Loss: {test_loss:.4f} | Test: {test_acc*100:.2f}% | "
                      f"Time: {self.Budget[-1]:.1f}s")

            self._log_wandb(round_idx, test_acc, test_loss)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        if self.rs_test_acc:
            print("\nBest accuracy: %.4f" % max(self.rs_test_acc))
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

    def check_done(self, acc_lss, top_cnt=100):
        for acc_ls in acc_lss:
            if len(acc_ls) < top_cnt:
                return False
            recent = acc_ls[-top_cnt:]
            history = acc_ls[:-top_cnt] + [0.0]
            if max(recent) <= max(history):
                return True
        return False