#!/usr/bin/env python
"""
FedVKD 项目主入口（修复版）

修复点：
1. wandb 改为 lazy import，未安装也能跑非 wandb 实验
2. 所有命令行 bool 参数改用 str2bool（避免 Python type=bool 经典坑）
3. 抽出 _split_classifier_attr / _wrap_with_base_head，自动适配
   不同模型的分类器命名（fc / classifier / linear / head）
4. 内层循环变量改名 cls，避免覆盖外层 run_idx
5. FedSAM 分支删掉对未定义 args.momentum 的 print
6. FedVKD 默认值与 client 修订版保持一致（alpha_0=0.5, ema_mu=0.5）
"""
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import torch.nn as nn

from flcore.servers.serveravg import FedAvg
from flcore.servers.servervls import FedVLS
from flcore.servers.servermr import FedMR
from flcore.servers.serverntd import FedNTD
from flcore.servers.serversam import FedSAM
from flcore.servers.serverlogitcal import FedLogitCal
from flcore.servers.serverrs import FedRS
from flcore.servers.serverexp import FedEXP
from flcore.servers.serverprox import FedProx
from flcore.servers.servermoon import MOON
from flcore.servers.servervkd import FedVKD
from flcore.servers.serverccvr import CCVR

from flcore.trainmodel.models import *
from flcore.trainmodel.resnetcifar import *
from flcore.trainmodel.mobilenetv2 import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter
from data.pacs_dataset import *
from data.meta_dataset import *
from data.generate_mnist import *
from dataset_utils import partition_data, get_dataloader

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(10)

# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
emb_dim = 32


# ====================================================================
# 工具函数
# ====================================================================

def str2bool(v):
    """argparse 专用 bool 解析器。
    Python 内置 bool('False') 会得到 True（任何非空字符串都是真），
    所以所有从命令行读 bool 的参数都要用这个函数当 type。
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', '1', 'y'):
            return True
        if v.lower() in ('no', 'false', 'f', '0', 'n'):
            return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v!r}')


def _split_classifier_attr(model):
    """寻找模型的最后一层分类器叫什么。
    - DNN / ResNet  ->  fc
    - MobileNetV2   ->  classifier
    - 其他可能叫 linear / head
    """
    for attr in ('fc', 'classifier', 'linear', 'head'):
        if hasattr(model, attr):
            return attr
    raise AttributeError(
        f"No classifier head found in {type(model).__name__}. "
        f"Tried: fc / classifier / linear / head"
    )


def _wrap_with_base_head(model):
    """把模型拆成 (base, head) 的 BaseHeadSplit 结构，
    无论分类器叫 fc / classifier / linear / head 都能正确处理。
    """
    attr = _split_classifier_attr(model)
    head = copy.deepcopy(getattr(model, attr))
    setattr(model, attr, nn.Identity())
    return BaseHeadSplit(model, head), head


# ====================================================================
# 主流程
# ====================================================================

def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model
    args.model_name = args.model

    for run_idx in range(args.prev, args.times):
        print(f"\n============= Running time: {run_idx}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # ---------- 1. 构造模型 ----------
        if model_str == "dnn":
            if "mnist" in args.dataset:
                args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
            elif "cifar10" in args.dataset:
                args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet18":
            args.model = torchvision.models.resnet18(
                pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet32":
            args.model = resnet32(num_classes=args.num_classes).to(args.device)

        elif model_str == "mobilenetv2":
            args.model = mobilenetv2(num_classes=args.num_classes).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # ---------- 2. 构造数据 ----------
        if args.dataset == 'mnist':
            party2loaders, global_train_dl, test_dl = generate_mnist(
                args.datadir, args.num_classes, args.num_clients,
                niid=True, balance=False, partition=args.partition, alpha=args.alpha
            )
        else:
            party2dataidx = partition_data(
                args.dataset, args.datadir, args.partition,
                args.num_clients, alpha=args.alpha
            )

            party2loaders = {}
            party2loaders_ds = {}
            datadistribution = np.zeros((args.num_clients, args.num_classes, 2))

            for party_id in range(args.num_clients):
                train_dl_local, _, train_ds_local, _ = get_dataloader(
                    args, args.dataset, args.datadir,
                    args.batch_size, args.batch_size, party2dataidx[party_id]
                )
                party2loaders[party_id] = train_dl_local
                party2loaders_ds[party_id] = train_ds_local

                # 注意：内层循环变量改名 cls，避免覆盖外层 run_idx
                for cls in range(args.num_classes):
                    datadistribution[party_id][cls][0] = cls

                all_labels = np.empty((0,), dtype=np.int64)
                for data, targets in party2loaders[party_id]:
                    labels = targets.numpy()
                    all_labels = np.concatenate((all_labels, labels), axis=0)
                uniq_val, uniq_count = np.unique(all_labels, return_counts=True)
                for j, c in enumerate(uniq_val.tolist()):
                    datadistribution[party_id][c][1] = uniq_count[j]

            np.set_printoptions(threshold=np.inf)
            print(datadistribution)

            global_train_dl, test_dl, _, _ = get_dataloader(
                args, args.dataset, args.datadir,
                train_bs=args.batch_size, test_bs=args.batch_size
            )

        # ---------- 3. 选择算法并实例化 server ----------
        if args.algorithm == "FedAvg":
            args.model, args.head = _wrap_with_base_head(args.model)
            server = FedAvg(args, run_idx, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedVLS":
            args.model, args.head = _wrap_with_base_head(args.model)
            server = FedVLS(args, run_idx, party2loaders, global_train_dl, test_dl)

        elif args.algorithm in ["FedVKD", "FedVKDHeadCal"]:
            args.model, args.head = _wrap_with_base_head(args.model)
            if args.algorithm == "FedVKDHeadCal":
                args.use_headcal = True
            server = FedVKD(args, run_idx, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedMR":
            args.model, args.head = _wrap_with_base_head(args.model)
            server = FedMR(args, run_idx, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedNTD":
            args.model, args.head = _wrap_with_base_head(args.model)
            server = FedNTD(args, run_idx, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedLogitCal":
            args.model, args.head = _wrap_with_base_head(args.model)
            server = FedLogitCal(args, run_idx, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedSAM":
            server = FedSAM(args, run_idx, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedRS":
            server = FedRS(args, run_idx, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedEXP":
            server = FedEXP(args, run_idx, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "FedProx":
            server = FedProx(args, run_idx, party2loaders, global_train_dl, test_dl)

        elif args.algorithm == "MOON":
            args.model, args.head = _wrap_with_base_head(args.model)
            server = MOON(args, run_idx, party2loaders, global_train_dl, test_dl)
        
        elif args.algorithm == "CCVR":
            args.model, args.head = _wrap_with_base_head(args.model)
            server = CCVR(args, run_idx, party2loaders, global_train_dl, test_dl)

        else:
            raise NotImplementedError

        # ---------- 4. wandb 初始化（lazy import）----------
        if args.use_wandb:
            import wandb  # ★ lazy：未安装也能跑非 wandb 实验
            run_name = args.wandb_run_name or (
                f"{args.algorithm}_{args.dataset}_alpha{args.alpha}_E{args.local_epochs}"
            )
            # 过滤掉不可序列化的对象（模型、head、device）
            config_dict = {
                k: v for k, v in vars(args).items()
                if not isinstance(v, (torch.nn.Module, torch.device))
            }
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=config_dict,
                reinit=True,
            )

        # ---------- 5. 训练 ----------
        server.train()

        if args.use_wandb:
            import wandb
            wandb.finish()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    print("All done!")
    reporter.report()


# ====================================================================
# 入口
# ====================================================================

if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()

    # ===== general =====
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="dnn",
                        choices=["dnn", "resnet18", "resnet32", "mobilenetv2"])
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ed', "--weight_decay", type=float, default=1e-5,
                        help="weight decay during local training")
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=str2bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=2,
                        help="Total number of clients")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-ab', "--auto_break", type=str2bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=str2bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-dd', '--datadir', type=str, required=False, default="./data/",
                        help="Data directory")

    # ===== practical =====
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    # ===== FedProx =====
    parser.add_argument('-bt', "--beta", type=float, default=0.005,
                        help="Average moving parameter for pFedMe, "
                             "Second learning rate of Per-FedAvg, "
                             "or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.001,
                        help="Proximal rate for FedProx")

    # ===== MOON =====
    parser.add_argument('-pro_d', "--proj_dim", type=int, default=256,
                        help='projection dimension of the projector')
    parser.add_argument('-tem', "--temperature", type=float, default=0.5,
                        help='the temperature parameter for contrastive loss')
    parser.add_argument('-use_prod', "--use_proj_head", type=str2bool, default=True,
                        help='whether to use projection head')

    # ===== non-iid =====
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    parser.add_argument('-partition', '--partition', type=str, default='noniid',
                        help='the data partitioning strategy')
    parser.add_argument('-aug', '--auto_aug', type=str2bool, default=True,
                        help='whether to apply auto augmentation')

    parser.add_argument('-tau', "--tau", type=float, default=0.001,
                        help='tau introduced in FedAdam paper. '
                             'Essentially, this hyper-parameter provides '
                             'numeric protection for second-order momentum')

    # ===== FedSAM =====
    parser.add_argument('-rho', "--rho", type=float, default=1.0,
                        help="rho hyper-parameter for sam")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="momentum for FedSAM base optimizer")

    # ===== FedLogitCal =====
    parser.add_argument('-cal_tem', "--calibration_temp", type=float, default=0.1,
                        help='calibration temperature')

    # ===== FedRS =====
    parser.add_argument('-rs', "--restricted_strength", type=float, default=0.5,
                        help='hyper-parameter for restricted strength')

    # ===== FedExp =====
    parser.add_argument('-eps', "--eps", type=float, default=1e-3,
                        help='epsilon of the FedExp algorithm')

    # ===== FedVKD 专用 =====
    parser.add_argument('--temperature_kd', type=float, default=3.0,
                        help='FedVKD: 蒸馏温度 T')
    parser.add_argument('--gamma_schedule', type=float, default=1.5,
                        help='FedVKD: 渐进调度曲线指数')
    parser.add_argument('--beta_vkd', type=float, default=0.7,
                        help='FedVKD: logit蒸馏 vs feature对齐 的权重')
    parser.add_argument('--alpha_0', type=float, default=0.5,
                        help='FedVKD: 基础蒸馏强度（推荐 0.5）')
    parser.add_argument('--ema_mu', type=float, default=0.5,
                        help='FedVKD: 脆弱度EMA平滑系数（推荐 0.5）')
    parser.add_argument('--warmup_rounds', type=int, default=10,
                        help='FedVKD: 前若干轮纯 CE，避免蒸馏随机模型')
    parser.add_argument('--vuln_threshold', type=float, default=0.05,
                        help='FedVKD: 脆弱度激活阈值（绝对概率差）')

    # ===== HeadCal =====
    parser.add_argument('--use_headcal', type=str2bool, default=False,
                        help='Enable HeadCal after FedVKD aggregation')
    parser.add_argument('--headcal_start_round', type=int, default=-1,
                        help='Start HeadCal after this round. -1 means use warmup_rounds.')
    parser.add_argument('--headcal_interval', type=int, default=5,
                        help='Run HeadCal every N rounds')
    parser.add_argument('--headcal_epochs', type=int, default=5,
                        help='HeadCal head-only training epochs')
    parser.add_argument('--headcal_lr', type=float, default=0.01,
                        help='HeadCal learning rate')
    parser.add_argument('--headcal_weight_decay', type=float, default=0.0,
                        help='HeadCal head optimizer weight decay')
    parser.add_argument('--headcal_samples_per_class', type=int, default=256,
                        help='Balanced feature samples per class for HeadCal')
    parser.add_argument('--headcal_batch_size', type=int, default=256,
                        help='HeadCal mini-batch size')
    parser.add_argument('--save_best', type=str2bool, default=True,
                        help='Whether to save best checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save best checkpoints')

    # ===== WandB =====
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='是否启用 wandb 日志')
    parser.add_argument('--wandb_project', type=str, default='FedVKD',
                        help='wandb 项目名')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='wandb 团队/用户名')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='run 名称（留空自动生成）')

    args = parser.parse_args()

    if args.algorithm == "FedVKDHeadCal":
        args.use_headcal = True

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    # ---------- 配置打印 ----------
    print("=" * 50)
    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Weight decay: {}".format(args.weight_decay))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("weight_decay: {}".format(args.weight_decay))
    print("noniid level: {}".format(args.alpha))
    print("auto_aug or not : {}".format(args.auto_aug))

    if args.algorithm == "FedProx":
        print("the coefficient of prox loss : {}".format(args.mu))
    elif args.algorithm == "MOON":
        print("the coefficient of moon loss : {}".format(args.mu))
        print("the projection dimension of the projector : {}".format(args.proj_dim))
        print("the temperature parameter for contrastive loss : {}".format(args.temperature))
        print("whether to use projection head : {}".format(args.use_proj_head))
    elif args.algorithm == "FedSAM":
        print("rho : {}".format(args.rho))
    elif args.algorithm == "FedLogitCal":
        print("calibration_temp : {}".format(args.calibration_temp))
    elif args.algorithm == "FedRS":
        print("restricted_strength : {}".format(args.restricted_strength))
    elif args.algorithm == "FedEXP":
        print("eps : {}".format(args.eps))
    elif args.algorithm == "FedNTD":
        print("the coefficient of NTD loss : {}".format(args.beta))
    elif args.algorithm == "FedMR":
        print("the coefficient of deco loss : {}".format(args.mu))
    elif args.algorithm in ["FedVKD", "FedVKDHeadCal"]:
        print("alpha_0 : {}".format(args.alpha_0))
        print("temperature_kd : {}".format(args.temperature_kd))
        print("gamma_schedule : {}".format(args.gamma_schedule))
        print("beta_vkd : {}".format(args.beta_vkd))
        print("ema_mu : {}".format(args.ema_mu))
        print("warmup_rounds : {}".format(args.warmup_rounds))
        print("vuln_threshold : {}".format(args.vuln_threshold))
        print("use_headcal : {}".format(args.use_headcal))
        print("headcal_start_round : {}".format(args.headcal_start_round))
        print("headcal_interval : {}".format(args.headcal_interval))
        print("headcal_epochs : {}".format(args.headcal_epochs))
        print("headcal_lr : {}".format(args.headcal_lr))
        print("headcal_batch_size : {}".format(args.headcal_batch_size))
        print("headcal_weight_decay : {}".format(args.headcal_weight_decay))
        print("headcal_samples_per_class : {}".format(args.headcal_samples_per_class))
        print("save_best : {}".format(args.save_best))
        print("checkpoint_dir : {}".format(args.checkpoint_dir))
   
    print("=" * 50)

    run(args)

    current_struct_time1 = time.localtime(time.time())
    formatted_time1 = time.strftime("%Y-%m-%d %H:%M:%S", current_struct_time1)
    print(f"Finished at: {formatted_time1}")
