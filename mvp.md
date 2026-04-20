# MOGT 架构降维打击 —— GCP L4 云原生科研级验证任务清单 (顶会路线)

基于我们在《规范场流形上的耗散并行输运：打破Transformer瓶颈的 O(N) 终身学习架构》中提出的理论框架，现将项目的概念验证全面升级为精准对标 **ICLR / NeurIPS 顶会评审标准** 的严密实验矩阵。

本实验集已深度绑定于 **Google Cloud Platform (GCP) L4 GPU (24GB VRAM, Ada 架构)**。借助其充裕的显存与强悍的 FP8/BF16 Tensor Cores 吞吐能力，我们将以前所未有的激进尺度压榨 MOGT 架构的核心潜力，彻底撕碎现有 Transformer 的平方级显存壁垒，并直面诸如 Mamba 等前沿 SSM 架构的跨维打击。

---

## 🎯 实验一：内存墙剥离极限测试 (The O(N) Complexity Proof)
**实验目标**：不再局限于 32K 玩具尺度，而是利用 24G 显存将上下文推向极限。通过定量对比，证明在极其恐怖的超长背景（$L > 100K$）下，MOGT 的李群并行流形依然维持完美的水平线显存占用规律。

- [x] **1.1 FlashAttention-2 极限逼近**：即使是最优异的 FA2 在 $L=64K \to 128K$ 处也会暴露出算力/空间复合折损，作为靶系进行阻击测试。
- [x] **1.2 MOGT 大核压测**：驱动序列长度序列：`[8K, 32K, 64K, 128K, 256K]` 的极限推栈。调用底层 Python `torch.cuda.max_memory_allocated` 实现精确无水探针。 (已完成：L4 成功压通 128K，峰值 18.7GB)
- [x] **1.3 理论/实测时空双对数绘图**：在严谨的物理科学图表下（双轴对数坐标 `log-log plot`），直出包含计算崩溃红线的内存与 Wall-clock time 平坦曲线，建立 O(N) 的绝对防御护城河。

## 🎯 实验二：莫尔斯流形动力学与“零伪造”持续学习 (Overcoming Catastrophic Forgetting)
**实验目标**：绝不断崖式退缩，通过完全手写的原生优化器，展示基于 Onsager-Machlup 热力学作用量的优化更新规则，对灾难性遗忘的拓扑结构护盾效应。

- [x] **2.1 `MOGTOptimizer` 纯血组装**：不依赖 PyTorch 的高频补丁，重写底层 optimizer class，实现：
  - (a) 保存基准任务的历史 Fisher 信息或降维主成分；
  - (b) 在每次 Forward 后，执行新梯度的**零曲率正交投影 ($\mathcal{P}_{\mathcal{M}^\perp}$)**；
  - (c) 当偏离能量盆地底端时施加排斥拉力泛函惩罚。
- [x] **2.2 Sequential Split-CIFAR / Split-Language 漂移环境建模**：构造多阶段强排他性（Non-i.i.d）序列流源。
- [x] **2.3 史诗级基线同框对决**：无任何修饰地拉入 `Vanilla AdamW`, `EWC (Elastic Weight Consolidation)` 进行跨任务测试。
- [x] **2.4 保留率斜率评估**：要求证明通过热力学阻尼保护的 MOGT 能够在吞噬新任务领域知识时，历史 Test Accuracy 曲线呈现极其坚挺的“平台状结构”，而非指数级塌陷。

## 🎯 实验三：“大海捞针”（Passkey Retrieval）极限信息流穿透
**实验目标**：这是对无 Attention、极度强调“单向前向因果并行扫描”网络的最无情打击。我们需要正面证明：经过十几万次非交换矩阵级联的李群空间后，特定位置的高细粒度隐含义能否穿越至最后不被高频噪声磨散。

- [x] **3.1 L4 级广袤背景海床建造**：利用 GCP 生成高达 `L=128k` 的非规律虚假背景 Token 流。
- [x] **3.2 全流域植入探测**：编写 `benchmark_passkey.py` 自动化打标针脚。在序列早、中、晚期各段分别埋入唯一密文与 UUID。
- [x] **3.3 检索精准度定格与对数分析**：基于严格的检索返回 Token 与原特征计算欧几里得距离，测算在 `1k-128k` 全区间内，信息回拉精确率能够逼近并超越开源 Transformer 长视口微调模型。

## 🎯 实验四：Iso-FLOP 计算最优缩放定律测绘 (Micro-Scaling Rules of MOGT)
**实验目标**：在 MOGT 这样颠覆常识的架构下，传统的缩放定律是否起效是审稿人最急于验证的盲点。我们将严格遵循 DeepMind 提议的 Compute-Optimal 原理，定量画出一条绝对客观的曲线拟合。

- [x] **4.1 微型宇宙实验床搭建**：以完全均等的语料纯净度切分集 (如特定规模的 WikiText 子集) 为土壤。
- [x] **4.2 控制变量法扩容池**：配置严格参数等级梯次：`10M, 30M, 70M, 130M` 的单向扩展池，且**同步按比例放大对应的算力消耗步数 (Tokens Seen)**。
- [x] **4.3 双对数回归拟合推演 (Scipy 介入)**：运用严格的数学曲线拟合 `Loss(N) = a N^{-\alpha} + \epsilon`，力证 MOGT 是具备明确收敛幂律潜能的通用架构体。

## 🎯 实验五：130M 同体积主会级硬核对战 (The GCP L4 Meatgrinder)
**实验目标**：在理论彻底完备后，迎接近些年最具压迫力的纯血竞赛。通过绝对控制基座参数来证明在常规任务的基准下，MOGT 打出的子弹也同样具有杀伤力。

- [x] **5.1 基座模型搭建与开源竞品对齐**：构建并约束 `~130M MOGT (L=2048)` 以全面抗衡 HuggingFace 原生 `Mamba-130M` 以及传统霸主 `GPT-2 Small`。
- [x] **5.2 真实 WikiText-103 困惑度跑分**：全面应用 `train.py` 中的 AMP 混合精度与实时存档断点防御策略。目标取得与 Mamba 系列相若、低于自回归 Vanilla Transformer 的语言建模困惑度 (Perplexity, PPL)。
- [x] **5.3 WGMMA 指令利用率优化 (Triton 降维进阶)**：对核心前置算子 `torch.matrix_exp` 以及前缀扫描执行深底层的解剖，探索直接接入 `triton_scan.py` 以及 Triton Shared-Memory 排布的可能，防止其出现严重的高频损耗导致大批量吞吐数据不好看。

## 🛠️ GCP 云原生物理解图流
作为顶会基石的物理载体与执行手册：
- **资源锚定**：采用 Google Cloud Engine 实例化 Nvidia L4 容器（包含 CUDA Toolkit 12.x）。
- **极速代码流**：在 Mac/本地 VS Code 使用 `Remote-SSH` 全盘接管 L4 级数据区，无缝联动代码级排错。
- **实验存档约束**：严禁中途修改任何 Random Seed。严禁通过非对称的 Batch Size 人为制造吞吐量错觉。要求一切实验的复现脚本做到一键执行（Zero-Configuration Repro）。
