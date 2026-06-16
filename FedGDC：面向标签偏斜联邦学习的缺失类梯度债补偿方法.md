# FedGDC：面向标签偏斜联邦学习的缺失类梯度债补偿方法

> FedGDC = Federated Gradient-Debt Compensation
>
> 核心一句话：标签偏斜下，客户端本地缺失类不是简单“缺样本”，而是在每轮本地训练中持续欠下一笔没有被执行的正类分类头梯度。FedGDC 显式估计这笔“梯度债”，并在客户端本地训练阶段进行 head-only 补偿。

---

# 0. 方案定位

FedGDC 是在原 FedVKD / FedVKD-MIR 思路基础上进一步抽象出来的新方案。它不再把主贡献写成“基于原型生成 imagined feature”，而是把问题重新定义为：

> **Missing-class Gradient Debt：缺失类梯度债。**

在标签偏斜联邦学习中，客户端本地训练只看到自己的已有类别。对于本地缺失类别，交叉熵损失中没有正标签样本，因此对应分类头权重长期缺少正类梯度。FedGDC 的目标不是生成完整缺失类样本，也不是简单做服务端分类头校准，而是在客户端本地训练阶段，用全局类别原型估计并偿还这部分缺失的正类梯度。

这个方案的核心优势是：

1. 比“虚拟特征回放”更理论化；
2. 比“原型增强”更容易和 FedProto / CCVR / FedExIT 切开；
3. 比“蒸馏保留空类知识”更直接，因为它显式补分类头正类梯度；
4. 实现成本低，第一版可以用 prototype CE loss 快速验证。

---

# 1. 背景与问题

## 1.1 标签偏斜联邦学习

设联邦系统包含 \(K\) 个客户端，全局类别集合为：

\[
\mathcal{C}=\{1,2,\dots,C\}
\]

客户端 \(k\) 的本地数据集为 \(D_k\)，本地类别集合为：

\[
\mathcal{Y}_k=\{c\mid n_{k,c}>0\}
\]

当：

\[
n_{k,c}=0
\]

类别 \(c\) 对客户端 \(k\) 而言是本地缺失类。进一步地，当：

\[
0<n_{k,c}<N_{min}
\]

该类虽然有少量样本，但统计高度不可靠，本文称为 effectively missing class。默认：

\[
N_{min}=3
\]

标签偏斜下，FedAvg 容易出现本地分类头偏置。客户端只优化本地已有类别，分类头会向本地优势类倾斜；对于本地缺失类，模型没有机会在本地看到其正标签监督。

## 1.2 已有方法的不足

已有方法通常从以下角度处理标签偏斜：

1. **FedRS / FedLC**：调整 softmax 或 logits，缓解局部类别偏置；
2. **FedVLS / FedNTD**：通过知识蒸馏保留空类或非目标类知识；
3. **FedProto**：通过全局类别原型约束本地特征学习；
4. **CCVR / HeadCal**：在服务端使用虚拟表征或统计信息校准分类头；
5. **FedCPD / class proxy 方法**：区分 observed / missing class proxy 的聚合方式。

这些方法有效，但仍存在一个共同空白：

> 它们多数没有把本地缺失类问题显式刻画为“正类分类头梯度长期缺席”的优化问题。

FedGDC 的切入点正是这里。

---

# 2. 核心观察：缺失类不是缺数据，而是欠梯度

全局模型拆分为特征提取器和分类头：

\[
F_\theta(x)=h_\phi(f_\omega(x))
\]

对线性分类头，类别 \(c\) 的 logit 为：

\[
o_c=w_c^\top z+b_c
\]

其中：

\[
z=f_\omega(x)
\]

交叉熵对类别权重 \(w_c\) 的梯度为：

\[
\frac{\partial \mathcal{L}_{CE}}{\partial w_c}
=
(p_c-\mathbf{1}[y=c])z
\]

如果客户端 \(k\) 没有类别 \(c\) 的样本，则本地训练中永远没有 \(\mathbf{1}[y=c]=1\) 的样本项。于是，类别 \(c\) 的分类头权重 \(w_c\) 在客户端 \(k\) 上缺少如下正类更新趋势：

\[
-(1-p_c)z
\]

这就是缺失类梯度债的来源。

可以用一句话理解：

> 对缺失类而言，本地训练每进行一轮，分类头都少执行了一次本该把 \(w_c\) 拉向类别语义中心的更新。

因此，标签偏斜不只是数据分布不均，也不是单纯 logits 过置信；它还会造成一种累积性的优化缺口：

\[
\text{Missing-class Gradient Debt}
\]

FedGDC 的目标就是估计并偿还这笔债。

---

# 3. 方法总览

FedGDC 包含四个核心模块：

1. **Global Prototype Bank**：服务器维护当前轮全局类别原型；
2. **Local Gradient Debt Estimation**：客户端估计每个缺失类的梯度债；
3. **Server-side Debt Ledger**：服务器维护类别级全局债务账本；
4. **Debt-aware Head Compensation**：客户端对高债务类别进行 head-only 补偿。

整体逻辑为：

```text
全局原型提供类别语义中心
        ↓
客户端用当前 head 评估自己是否识别该原型
        ↓
若类别本地缺失且 head 对该类原型不自信，则产生梯度债
        ↓
服务器维护类别级长期债务账本
        ↓
客户端优先偿还“本地缺 + 全局也长期欠”的类别
```

---

# 4. Global Prototype Bank

## 4.1 原型统计

在每轮通信中，客户端基于当前全局模型提取本地特征，并上传安全聚合所需统计量。

类别 \(c\) 的全局样本数为：

\[
N_c^{(t)}=\sum_{k\in\mathcal{S}_t} n_{k,c}
\]

类别原型为：

\[
\mu_c^{(t)}=
\frac{
\sum_{k\in\mathcal{S}_t} n_{k,c}\mu_{k,c}^{(t)}
}{
N_c^{(t)}+\epsilon
}
\]

其中 \(\mathcal{S}_t\) 是第 \(t\) 轮参与训练的客户端集合。

## 4.2 原型可靠性

如果某类本轮参与样本过少，原型可能不可靠。定义：

\[
r_c^{(t)}=
\min\left(
1,
\frac{N_c^{(t)}}{\lambda_r \bar N^{(t)}+\epsilon}
\right)
\]

其中：

\[
\bar N^{(t)}=\frac{1}{C}\sum_{c=1}^{C}N_c^{(t)}
\]

若 \(N_c^{(t)}=0\)，则：

\[
r_c^{(t)}=0
\]

该类本轮不参与梯度债补偿。

## 4.3 轮次对齐原则

客户端第 \(t\) 轮用于 GDC 的 prototype bank 必须和当前下发模型处于同一特征空间。不能把旧模型下统计出的原型直接用于新模型空间。

第一版实现可采用：

```text
第 t 轮客户端使用 θ^(t) 和 B^(t)
客户端训练并上传统计
服务器聚合得到 θ^(t+1)
服务器构建下一轮 B^(t+1)
```

不建议第一版对全局原型做跨轮 EMA。EMA 可以作为消融，而不是默认主方法。

---

# 5. Local Gradient Debt Estimation

FedGDC 的关键是定义客户端 \(k\) 对类别 \(c\) 的梯度债：

\[
D_{k,c}^{(t)}
=
s_{k,c}^{(t)}
\cdot
r_c^{(t)}
\cdot
\Delta_{k,c}^{(t)}
\]

其中三项分别回答三个问题：

| 因子 | 问题 | 作用 |
| --- | --- | --- |
| \(s_{k,c}\) | 客户端是否缺这个类 | 只补真正缺失或稀缺的类别 |
| \(r_c\) | 服务器原型是否可信 | 防止错误原型误导补偿 |
| \(\Delta_{k,c}\) | 当前 head 是否欠拟合该类 | 只补模型确实不认识的类别 |

## 5.1 数据缺失度

定义客户端 \(k\) 对类别 \(c\) 的缺失度：

\[
s_{k,c}^{(t)}=
\begin{cases}
1, & n_{k,c}<N_{min} \\
\max(0,1-\frac{n_{k,c}}{\bar n_c}), & n_{k,c}\ge N_{min}\ \text{and}\ n_{k,c}<\bar n_c \\
0, & n_{k,c}\ge \bar n_c
\end{cases}
\]

其中 \(\bar n_c\) 是类别 \(c\) 在所有客户端上的平均样本数。

解释：

- 完全缺失类：\(s=1\)；
- 极少样本类：\(s=1\)，按 effectively missing 处理；
- 少于平均水平的弱类：\(0<s<1\)；
- 充分覆盖类：\(s=0\)。

## 5.2 Head 欠拟合程度

用当前客户端分类头对全局原型的预测来估计欠拟合程度。

将类别原型送入当前 head：

\[
p_{k,c}^{(t)}(\mu_c)=softmax(h_{\phi_k}(\mu_c))_c
\]

定义：

\[
\Delta_{k,c}^{(t)}=1-p_{k,c}^{(t)}(\mu_c)
\]

如果客户端 head 已经能把全局类别原型 \(\mu_c\) 识别为类别 \(c\)，则 \(p_c(\mu_c)\) 高，债务小；反之，债务大。

这一步非常关键，因为它把 FedGDC 和简单“缺哪个类就补哪个类”区分开：

> FedGDC 只补“本地缺失且当前 head 对其语义中心不自信”的类别。

## 5.3 梯度债定义

最终本地梯度债为：

\[
D_{k,c}^{(t)}
=
s_{k,c}^{(t)}r_c^{(t)}(1-p_{k,c}^{(t)}(\mu_c))
\]

若 \(D_{k,c}=0\)，该类别不需要补偿。

---

# 6. Server-side Debt Ledger

只看当前轮本地债务可能不稳定。某些类别可能长期被多数客户端缺失，导致它们在全局训练中持续欠补偿。为此，FedGDC 引入服务器端债务账本。

## 6.1 类别级全局债务

客户端上传本地债务摘要：

\[
\bar D_c^{(t)}=
\frac{1}{|\mathcal{S}_t|}
\sum_{k\in\mathcal{S}_t}D_{k,c}^{(t)}
\]

服务器维护 EMA 债务账本：

\[
L_c^{(t)}=ho L_c^{(t-1)}+(1-\rho)\bar D_c^{(t)}
\]

默认：

\[
\rho=0.9
\]

其中 \(L_c^{(t)}\) 表示类别 \(c\) 在全局训练中的长期梯度债。

## 6.2 为什么需要账本

没有账本时，客户端只根据本地状态补偿，容易出现两个问题：

1. 某些类别本轮债务低，但长期被欠补偿；
2. 某些类别局部债务高，但全局已经被充分训练。

债务账本让客户端优先补：

> 本地缺失，同时全局长期欠补偿的类别。

这也是 FedGDC 区别于普通 prototype replay 的核心创新之一。

---

# 7. Debt-aware Class Selection

客户端 \(k\) 对类别 \(c\) 的最终补偿分数为：

\[
q_{k,c}^{(t)}
=
D_{k,c}^{(t)}
\cdot
\tilde L_c^{(t)}
\]

其中：

\[
\tilde L_c^{(t)}=
\frac{L_c^{(t)}}{\frac{1}{C}\sum_{j=1}^{C}L_j^{(t)}+\epsilon}
\]

客户端选择：

\[
\mathcal{C}_{k}^{GDC}=TopM(q_{k,c}^{(t)})
\]

只考虑：

\[
q_{k,c}^{(t)}>0
\]

默认：

\[
M_k=\min(3, |\{c:q_{k,c}>0\}|)
\]

若 \(M_k=0\)，直接关闭 GDC。

---

# 8. Debt-aware Head Compensation

FedGDC 有两种实现方式。

第一版建议先用 **loss 形式**，稳定、简单、容易接入现有训练代码。

## 8.1 Loss 形式

对选中的高债务类别，构造：

\[
\mathcal{L}_{GDC}
=
\frac{1}{M_k}
\sum_{c\in\mathcal{C}_{k}^{GDC}}
\bar q_{k,c}^{(t)}
CE(h_{\phi_k}(detach(\mu_c^{(t)})),c)
\]

其中：

\[
\bar q_{k,c}^{(t)}=
\frac{q_{k,c}^{(t)}}{rac{1}{M_k}\sum_{j\in\mathcal{C}_{k}^{GDC}}q_{k,j}^{(t)}+\epsilon}
\]

该 loss 的含义是：对当前客户端欠债最严重的缺失类，在类别原型位置补一个正类训练信号。

工程实现：

```python
mu = proto_bank[c].detach()
logits = model.head(mu)
loss_gdc = ce(logits, target=c)
```

必须保证：

```python
mu = mu.detach()
```

GDC 只更新分类头，不更新 backbone。

## 8.2 直接 head update 形式

也可以直接构造近似梯度补偿：

\[
w_c \leftarrow
w_c + \eta_g \bar q_{k,c}(1-p_c(\mu_c))\hat\mu_c
\]

其中：

\[
\hat\mu_c=\frac{\mu_c}{\|\mu_c\|_2+\epsilon}
\]

这个形式更像“显式还债”，但第一版不建议先做，因为直接改参数更容易引入工程不稳定。可以作为后续增强或附录实验。

## 8.3 总损失

客户端总损失为：

\[
\mathcal{L}
=
\mathcal{L}_{CE}
+
\alpha_t\mathcal{L}_{VKD}
+
\gamma_t\mathcal{L}_{GDC}
\]

其中：

\[
\alpha_t=\alpha_0\cdot progress(t)
\]

\[
\gamma_t=\gamma_0\cdot progress(t)
\]

默认第一版：

\[
\gamma_0=\alpha_0
\]

如果暂时不使用 VKD，则：

\[
\alpha_t=0
\]

FedGDC 可以独立于 VKD 存在，也可以和 VKD 组成完整框架。

---

# 9. 算法流程

## 9.1 Server Side

```text
Initialize global model θ
Initialize prototype bank B = None
Initialize debt ledger L_c = 1 for each class c

for each communication round t:

    Select clients S_t

    Send θ, prototype bank B, and debt ledger L to selected clients

    for each client k in S_t:
        θ_k, proto_stats_k, debt_stats_k = ClientUpdate(k, θ, B, L)

    θ = WeightedAverage({θ_k})

    Securely aggregate class-wise prototype statistics
    for each class c:
        compute N_c
        if N_c == 0:
            mark prototype invalid
            set r_c = 0
        else:
            compute μ_c
            compute r_c

    Aggregate client debt summaries:
        D_bar_c = mean_k D_{k,c}

    Update debt ledger:
        L_c = ρ L_c + (1-ρ) D_bar_c

    Build prototype bank for next round
    Evaluate global model
```

## 9.2 Client Side

```text
ClientUpdate(k, θ, B, L):

    Load global model θ

    for each local epoch:
        for each batch (x, y):
            compute CE loss on real local data
            compute optional VKD loss

            if prototype bank B is available:
                for each valid class c:
                    compute scarcity s_{k,c}
                    compute p_c(μ_c) using local head
                    compute local debt D_{k,c} = s * r * (1 - p_c)
                    compute q_{k,c} = D_{k,c} * normalized L_c

                select Top-M debt classes

                if selected set not empty:
                    compute GDC loss on selected prototypes
                else:
                    GDC loss = 0
            else:
                GDC loss = 0

            total loss = CE + alpha * VKD + gamma * GDC
            update model

    submit prototype statistics
    submit debt statistics
    return updated model
```

---

# 10. 与已有工作的边界

## 10.1 vs CCVR / HeadCal

CCVR 和 HeadCal 主要是服务端 post-hoc classifier calibration。它们在全局模型训练后，用虚拟表征或统计信息校准分类头。

FedGDC 不同：

1. 发生在客户端本地训练阶段；
2. 按客户端缺失情况选择类别；
3. 目标是偿还缺失类正类梯度债；
4. 可以与 HeadCal 叠加，而不是替代。

## 10.2 vs FedVLS

FedVLS 关注 vacant classes，通过 vacant-class distillation 和 logit suppression 保留空类知识。

FedGDC 不同：

1. FedVLS 主要是 soft distillation / logits 约束；
2. FedGDC 显式构造缺失类正类分类头梯度；
3. FedGDC 有 debt ledger，可识别长期欠补偿类别。

## 10.3 vs FedLC / FedRS

FedLC / FedRS 调整本地 logits 或 softmax，使缺失类权重不被错误更新。

FedGDC 不同：

1. FedLC / FedRS 偏向“少伤害”；
2. FedGDC 偏向“主动补偿”；
3. FedGDC 明确提供缺失类正类 head supervision。

## 10.4 vs FedProto

FedProto 使用全局原型正则化本地特征，使本地原型靠近全局原型。

FedGDC 不同：

1. FedProto 主要约束 feature extractor；
2. FedGDC 主要补偿 classifier head；
3. FedGDC 不要求缺失类有本地样本来形成本地原型；
4. FedGDC 的核心变量是梯度债，而不是原型距离。

## 10.5 vs FedExIT / synthetic embedding fine-tuning

FedExIT 这类方法可能使用 global prototypes 生成 balanced synthetic embeddings 来微调分类器偏置。

FedGDC 不同：

1. FedGDC 不做全局 balanced synthetic embedding；
2. FedGDC 是 client-conditioned 的，按每个客户端的缺失情况补偿；
3. FedGDC 引入 debt ledger，区分长期欠补偿类别；
4. FedGDC 的理论表述是 gradient debt compensation，而不是 synthetic embedding calibration。

---

# 11. 创新点总结

FedGDC 的创新可以压缩为三点：

## 11.1 Missing-class Gradient Debt

首次将标签偏斜下缺失类性能下降刻画为分类头正类梯度长期缺席导致的梯度债问题。

## 11.2 Debt Ledger

提出服务器端类别级债务账本，记录各类别在全局训练中长期欠补偿程度，避免只依赖当前轮局部状态。

## 11.3 Debt-aware Head Compensation

提出客户端条件化的 head-only 债务补偿机制，根据本地缺失度、原型可靠性、head 欠拟合程度和全局债务账本选择补偿类别。

一句话创新闭环：

> 缺什么梯度 → 估计谁欠债 → 记录全局长期债务 → 在本地 head-only 偿还。

---

# 12. 实验设计

## 12.1 数据集

建议按以下顺序推进：

1. MNIST / Fashion-MNIST：sanity check；
2. CIFAR-10：主实验；
3. CIFAR-100：扩展实验；
4. Tiny-ImageNet：如果前面结果足够好，再考虑。

第一版只做 CIFAR-10 即可。

## 12.2 Non-IID 设置

使用 Dirichlet 标签偏斜：

\[
\alpha\in\{0.1,0.3,0.5\}
\]

客户端数量：

\[
K\in\{10,20\}
\]

参与比例：

\[
join\_ratio\in\{1.0,0.5\}
\]

第一版先跑：

```text
Dataset: CIFAR-10
K: 10
Dirichlet alpha: 0.1
join_ratio: 1.0
seeds: 3
```

## 12.3 Baselines

外部 baseline：

1. FedAvg
2. FedProx
3. FedRS
4. FedLC
5. FedVLS
6. FedProto
7. CCVR / HeadCal

如果实现压力大，最低保留：

```text
FedAvg
FedRS
FedLC
FedVLS
HeadCal / CCVR
```

## 12.4 FedGDC 消融

| 变体 | 目的 |
| --- | --- |
| w/o GDC | 验证债务补偿是否有用 |
| local debt only | 只用 \(D_{k,c}\)，不用 ledger |
| global ledger only | 只用 \(L_c\)，不用本地 debt |
| random debt classes | 随机选缺失类，验证 debt selection |
| prototype CE replay | 普通原型回放，对比 GDC |
| w/o reliability | 去掉 \(r_c\)，验证原型可靠性 |
| w/o scarcity | 去掉 \(s_{k,c}\)，验证客户端缺失条件 |
| direct head update | 与 loss 形式比较 |
| GDC + VKD | 验证蒸馏和债务补偿互补 |
| GDC + HeadCal | 验证与服务端校准互补 |

## 12.5 评价指标

必须报告：

1. Overall accuracy
2. Macro accuracy
3. Worst-class accuracy
4. Missing-class accuracy
5. Tail-class accuracy
6. Per-class accuracy std
7. MIR/GDC activated class ratio
8. GDC loss curve
9. GDC/CE gradient ratio
10. Client training overhead
11. Communication overhead

其中最关键的是：

```text
missing-class accuracy
worst-class accuracy
macro accuracy
per-class accuracy std
```

如果这几个指标不明显提升，FedGDC 的故事就不成立。

---

# 13. 第一版超参数建议

| 超参 | 建议值 | 说明 |
| --- | --- | --- |
| \(T_{warm}\) | 5 rounds | 先建立原型语义 |
| \(N_{min}\) | 3 | 极少样本类视为 effectively missing |
| \(M_k\) | 3 | 每轮最多补 3 个高债务类别 |
| \(\rho\) | 0.9 | debt ledger EMA |
| \(\gamma_0\) | 0.1 或等于 \(\alpha_0\) | GDC loss 权重 |
| \(\lambda_r\) | 0.5 或 1.0 | 原型可靠性缩放 |
| warmup progress | linear | 平滑启用 GDC |

第一版建议：

\[
\gamma_t=\gamma_0\cdot \min(1,\frac{t-T_{warm}}{T_{ramp}})
\]

其中：

\[
T_{ramp}=10
\]

避免 GDC 在原型不稳定时突然介入。

---

# 14. 理想实验结果

FedGDC 最理想的结果形态不是 overall accuracy 暴涨，而是：

1. Overall accuracy 小幅提升或不下降；
2. Macro accuracy 明显提升；
3. Worst-class accuracy 明显提升；
4. Missing-class accuracy 明显提升；
5. Per-class accuracy std 下降；
6. GDC/CE gradient ratio 稳定；
7. 与 HeadCal / CCVR 有互补。

理想排序：

```text
FedGDC + HeadCal
>= FedGDC full
> local debt only
> prototype CE replay
> random debt classes
> w/o GDC
> FedAvg
```

如果 `prototype CE replay` 和 FedGDC full 差不多，说明 debt ledger 的贡献不明显；如果 `random debt classes` 差不多，说明 debt score 没有价值；如果 `HeadCal` 单独已经碾压 FedGDC，说明本地阶段补偿必要性不足。

---

# 15. 风险与止损标准

## 15.1 主要风险

1. 原型质量差，导致债务估计错误；
2. Head 对原型的置信度不稳定，\(\Delta_{k,c}\) 噪声大；
3. Debt ledger 过度平滑，响应变慢；
4. GDC 与 CE 梯度冲突，损害本地已有类；
5. HeadCal 已经解决大部分分类头偏置，GDC 增益不明显。

## 15.2 止损标准

如果第一版 CIFAR-10 α=0.1 出现以下情况，建议停止或大改：

1. Missing-class accuracy 提升小于 2%；
2. Worst-class accuracy 没有提升；
3. Overall accuracy 明显下降超过 1%；
4. full GDC 与 random debt classes 差不多；
5. full GDC 与 prototype CE replay 差不多；
6. GDC + HeadCal 没有任何互补。

## 15.3 继续推进标准

如果满足以下情况，值得继续扩实验：

1. Missing-class accuracy 提升 3% 以上；
2. Worst-class accuracy 提升 2% 以上；
3. Overall accuracy 持平或小涨；
4. full GDC 明显优于 random debt classes；
5. full GDC 明显优于 prototype CE replay；
6. GDC + HeadCal 仍能小幅提升。

---

# 16. 论文写法建议

论文标题建议：

> FedGDC: Gradient-Debt Compensation for Missing Classes in Label-Skewed Federated Learning

中文标题：

> FedGDC：面向标签偏斜联邦学习的缺失类梯度债补偿方法

摘要核心句：

> We identify missing-class degradation under label-skewed federated learning as a gradient-debt problem: absent classes do not receive positive classifier gradients during local training. To address this, we propose FedGDC, which estimates client-wise missing-class gradient debt using global prototypes and maintains a server-side debt ledger to guide debt-aware head-only compensation.

中文：

> 本文将标签偏斜下缺失类性能下降重新刻画为梯度债问题：本地缺失类别在客户端训练中长期缺少正类分类头梯度。为此，提出 FedGDC，利用全局原型估计客户端缺失类梯度债，并在服务器端维护类别级债务账本，引导客户端执行债务感知的 head-only 补偿更新。

---

# 17. 与 FedVKD / FedVKD-MIR 的关系

如果保留 FedVKD 的蒸馏模块，可以把完整框架写成：

```text
FedGDC = VKD soft knowledge preservation + GDC missing-class gradient debt compensation
```

其中：

- VKD 负责保留全局 soft knowledge；
- GDC 负责补偿缺失类正向分类头梯度；
- 二者互补：一个保知识，一个还梯度债。

不要把论文写成：

```text
FedVKD + 一个 GDC 小模块
```

而要写成：

```text
FedGDC：一个以缺失类梯度债为核心的新框架
```

FedVKD 可以作为 ablation：

```text
FedGDC w/o GDC = distillation-only variant
```

这样更像完整论文。

---

# 18. 最终判断

FedGDC 相比 FedVKD-MIR 更适合作为论文主 idea。原因是：

1. 它有更清楚的问题重定义：missing-class gradient debt；
2. 它不依赖“生成特征”作为唯一卖点，能避开 CCVR / FedExIT 的部分重合；
3. 它有独立机制：debt ledger；
4. 它能设计强消融；
5. 第一版实现成本低。

一句话：

> FedVKD-MIR 是“用原型补缺失类特征”；FedGDC 是“定义并偿还缺失类梯度债”。后者更像论文。
