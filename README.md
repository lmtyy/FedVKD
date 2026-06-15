# FedVKD

FedVKD is a federated learning research codebase for label-skewed non-IID classification. The current implementation extends the original FedVLS-style baseline with **Data-Aware + Vulnerability-Aware Selective Knowledge Distillation** and an optional server-side **HeadCal** classifier-head calibration module.

The main entry is `system/main.py`.

## What This Repository Implements

### FedVKD

FedVKD targets the common non-IID failure mode where a client has missing or scarce classes and the global classifier becomes biased toward locally frequent classes.

The current FedVKD client implements:

- **Warmup training**: before `warmup_rounds`, clients train with CE only.
- **Data-aware class weights**: missing classes receive high distillation weight; scarce classes receive partial weight; sufficiently represented classes are not over-distilled.
- **Vulnerability-aware EMA**: each client keeps a class-wise vulnerability estimate from teacher/student probability gaps.
- **Selective KD**: effective distillation weights combine data scarcity and class vulnerability.
- **Dual KD branches**:
  - logit-level KD with `temperature_kd ** 2` scaling;
  - normalized feature alignment for feature-level distillation.
- **Base/head model split**: models are wrapped as `BaseHeadSplit`, so FedVKD can call `model.base(x)` and `model.head(feature)`.

### HeadCal

HeadCal is an optional server-side classifier-head calibration step for FedVKD.

It runs after server aggregation and before evaluation:

1. Copy the current global model.
2. Recalibrate BN statistics on the copy.
3. Freeze the feature extractor.
4. Extract class-balanced features from the global training loader.
5. Train only a copied classifier head.
6. Copy only the calibrated head back into `self.global_model.head`.

HeadCal does **not** update the backbone / feature extractor.

Enable it either by:

```bash
-algo FedVKDHeadCal
```

or by:

```bash
-algo FedVKD --use_headcal True
```

`FedVKDHeadCal` is only an alias for `FedVKD` with `use_headcal=True`.

### BN Recalibrated Evaluation

For FedAvg / FedVLS / FedVKD, evaluation uses a copied global model with BatchNorm recalibration. This avoids polluting the model that will be sent to clients in the next round.

Best checkpoint behavior in FedVKD:

- `model_state`: BN-recalibrated `eval_model.state_dict()` that produced the reported best accuracy.
- `raw_model_state`: non-recalibrated aggregated `global_model.state_dict()` for later analysis or continued training.

## Installation

Create the environment from the provided file:

```bash
conda env create -f env_cuda_latest.yaml
conda activate fl
```

If you use another environment, make sure these packages are available:

- `torch`
- `torchvision`
- `numpy`
- `scipy`
- `scikit-learn`
- `h5py`
- `calmsize`

## Datasets

Supported datasets in the current training entry:

| Dataset | Notes |
|---|---|
| `mnist` | Downloaded automatically by torchvision. |
| `cifar10` | Downloaded automatically by torchvision. |
| `cifar100` | Downloaded automatically by torchvision. |
| `tinyimagenet` | Requires manual download and preprocessing. |

Default data directory:

```text
./data/
```

For TinyImageNet, place the dataset under `data/` and follow the original TinyImageNet preprocessing convention used by the repository.

## Model Choices

`main.py` currently supports:

| Argument | Model |
|---|---|
| `-m dnn` | DNN for MNIST-style inputs. |
| `-m resnet18` | torchvision ResNet-18. |
| `-m resnet32` | CIFAR ResNet-32. |
| `-m mobilenetv2` | CIFAR-style MobileNetV2. |

The default is `dnn`.

## Algorithms

The repository still contains several baseline / comparison algorithms:

- `FedAvg`
- `FedVLS`
- `FedVKD`
- `FedVKDHeadCal`
- `FedMR`
- `FedNTD`
- `FedLogitCal`
- `FedSAM`
- `FedRS`
- `FedEXP`
- `FedProx`
- `MOON`
- `CCVR`

The most actively maintained path in this project is `FedVKD` / `FedVKDHeadCal`.

## Common Arguments

| Argument | Meaning | Default |
|---|---|---|
| `-dev`, `--device` | Device type: `cpu` or `cuda`. | `cuda` |
| `-did`, `--device_id` | CUDA device id. | `0` |
| `-data`, `--dataset` | Dataset name. | `mnist` |
| `-m`, `--model` | Model architecture. | `dnn` |
| `-nb`, `--num_classes` | Number of classes. | `10` |
| `-lbs`, `--batch_size` | Local batch size. | `10` |
| `-lr`, `--local_learning_rate` | Local learning rate. | `0.005` |
| `-ed`, `--weight_decay` | Weight decay. | `1e-5` |
| `-gr`, `--global_rounds` | Number of communication rounds. | `100` |
| `-ls`, `--local_epochs` | Local epochs per round. | `1` |
| `-algo`, `--algorithm` | Algorithm name. | `FedAvg` |
| `-nc`, `--num_clients` | Total clients. | `2` |
| `-jr`, `--join_ratio` | Fraction of clients selected per round. | `1.0` |
| `-partition`, `--partition` | Data partition strategy. | `noniid` |
| `-al`, `--alpha` | Dirichlet coefficient for non-IID split. | `1.0` |
| `-aug`, `--auto_aug` | Whether to use AutoAugment. | `True` |
| `--momentum` | Momentum for FedSAM base optimizer. | `0.9` |

Boolean CLI options use robust string parsing, e.g. `True`, `False`, `1`, `0`, `yes`, `no`.

## FedVKD Arguments

| Argument | Meaning | Default |
|---|---|---|
| `--temperature_kd` | KD temperature `T`. | `3.0` |
| `--gamma_schedule` | Progressive KD schedule exponent. | `1.5` |
| `--beta_vkd` | Weight for logit KD; feature KD uses `1 - beta_vkd`. | `0.7` |
| `--alpha_0` | Base KD strength. | `0.5` |
| `--ema_mu` | EMA smoothing factor for class vulnerability. | `0.5` |
| `--warmup_rounds` | Rounds using CE only before KD enters the loss. | `10` |
| `--vuln_threshold` | Vulnerability threshold for stronger effective weights. | `0.05` |

During warmup, vulnerability can still be estimated, but KD loss is not added to the training objective.

## HeadCal Arguments

| Argument | Meaning | Default |
|---|---|---|
| `--use_headcal` | Enable server-side HeadCal. | `False` |
| `--headcal_start_round` | Start round. `-1` means use `warmup_rounds`. | `-1` |
| `--headcal_interval` | Run HeadCal every N rounds. | `5` |
| `--headcal_epochs` | Head-only calibration epochs. | `5` |
| `--headcal_lr` | HeadCal learning rate. | `0.01` |
| `--headcal_weight_decay` | HeadCal optimizer weight decay. | `0.0` |
| `--headcal_samples_per_class` | Balanced samples per class used for head training. | `256` |
| `--headcal_batch_size` | HeadCal mini-batch size. | `256` |

For memory protection, HeadCal keeps only a bounded number of candidate features per class before balanced sampling.

## Checkpoint and Logging

| Argument | Meaning | Default |
|---|---|---|
| `--save_best` | Save best FedVKD checkpoint. | `True` |
| `--checkpoint_dir` | Directory for best checkpoints. | `./checkpoints` |
| `--use_wandb` | Enable Weights & Biases logging. | `False` |
| `--wandb_project` | W&B project name. | `FedVKD` |
| `--wandb_entity` | W&B entity. | `None` |
| `--wandb_run_name` | W&B run name. | auto-generated |

`wandb` is lazily imported, so it is not required for normal non-W&B experiments.

## Quick Debug Commands

Run from the repository root:

```bash
cd system
```

### FedAvg smoke test on CPU

```bash
python main.py \
  -dev cpu \
  -data mnist \
  -m dnn \
  -algo FedAvg \
  -gr 1 \
  -nc 2 \
  -jr 1 \
  -lbs 8 \
  -ls 1
```

### FedVKD smoke test on CPU

```bash
python main.py \
  -dev cpu \
  -data mnist \
  -m dnn \
  -algo FedVKD \
  -gr 1 \
  -nc 2 \
  -jr 1 \
  -lbs 8 \
  -ls 1 \
  --save_best False
```

### FedVKD + HeadCal quick debug

```bash
python main.py \
  -dev cpu \
  -data mnist \
  -m dnn \
  -algo FedVKDHeadCal \
  -gr 2 \
  -nc 2 \
  -jr 1 \
  -lbs 8 \
  -ls 1 \
  --warmup_rounds 0 \
  --headcal_epochs 1 \
  --headcal_samples_per_class 32 \
  --save_best False
```

## Recommended CIFAR-10 Commands

### FedVKD baseline

```bash
cd system
python main.py \
  -data cifar10 \
  -m resnet32 \
  -algo FedVKD \
  -gr 100 \
  -nc 10 \
  -lbs 64 \
  -dev cuda \
  -did 0 \
  -al 0.1 \
  -partition noniid \
  -aug True \
  --warmup_rounds 10 \
  --alpha_0 0.5 \
  --temperature_kd 3.0 \
  --gamma_schedule 1.5 \
  --beta_vkd 0.7 \
  --ema_mu 0.5 \
  --vuln_threshold 0.05
```

### FedVKD + HeadCal

```bash
cd system
python main.py \
  -data cifar10 \
  -m resnet32 \
  -algo FedVKDHeadCal \
  -gr 100 \
  -nc 10 \
  -lbs 64 \
  -dev cuda \
  -did 0 \
  -al 0.1 \
  -partition noniid \
  -aug True \
  --warmup_rounds 10 \
  --alpha_0 0.5 \
  --temperature_kd 3.0 \
  --gamma_schedule 1.5 \
  --beta_vkd 0.7 \
  --ema_mu 0.5 \
  --vuln_threshold 0.05 \
  --headcal_start_round 10 \
  --headcal_interval 5 \
  --headcal_epochs 5 \
  --headcal_lr 0.01 \
  --headcal_samples_per_class 256 \
  --headcal_batch_size 256
```

## Expected Logs

FedVKD logs include:

- `alpha`: scheduled KD strength.
- `distill_cls`: classes with data-aware distillation weight.
- `effective_cls`: classes with effective vulnerability-aware distillation weight.
- `vuln_cls`: classes above vulnerability threshold.
- `kd_loss`, `logit_kd`, `feat_kd`: KD diagnostics.
- `vulnerability_mean`, `vulnerability_max`: vulnerability EMA diagnostics.

HeadCal logs include:

```text
[HeadCal] round=... ran=1 before=...% after=...% delta=...pt loss=... classes=... samples=...
```

Checkpoint logs include:

```text
[Checkpoint] New best ... saved BN-recalibrated eval model to ...
```

## Development Checks

Syntax check:

```bash
python -m py_compile \
  system/main.py \
  system/flcore/clients/clientvkd.py \
  system/flcore/servers/servervkd.py \
  system/flcore/clients/clientsam.py \
  system/flcore/servers/serversam.py
```

Confirm there are no hard-coded CUDA moves:

```bash
grep -R "\.cuda()" -n system --include="*.py"
```

This should print nothing.

## Notes and Known Caveats

- MNIST / CIFAR datasets may download automatically depending on torchvision availability and network access.
- TinyImageNet needs manual preparation.
- HeadCal is bounded by class-wise feature caps, but it can still add overhead on large datasets.
- Some legacy algorithms are kept for comparison and compatibility; the most actively maintained path is FedVKD / FedVKDHeadCal.

## Acknowledgement

This repository builds on ideas and code patterns from FedVLS, CCVR, and PFLlib-style federated learning implementations.
