# MOGT: Magnus-Onsager Gauge Transport

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)

## Strategic Roadmap

The project now has one strategic control document:

- [Top-tier Transformer disruption roadmap](./docs/TOP_TIER_TRANSFORMER_DISRUPTION_ROADMAP.md)
- [Project freeze / VM handoff](./docs/PROJECT_FREEZE_20260505.md)
- [New VM quickstart](./docs/NEW_VM_QUICKSTART.md)

Use it as the authority for what to build next, what claims are allowed, and how
the repo should be reorganized.

## Resume On A New VM

```bash
git clone https://github.com/lkcfqy/MOGT.git
cd MOGT
bash scripts/bootstrap_new_vm.sh
```

The default bootstrap installs core dependencies, validates stored experiment
reports, and regenerates the paper result snapshot. Optional CUDA-extension
baselines can be installed with `INSTALL_OPTIONAL_GPU=1`.

MOGT 是一个研究型非注意力语言模型原型。项目当前的核心目标不是一次性证明全部理论叙事，而是先把最关键的算子站稳：

`H_t = U_t @ H_{t-1} + V_t`

也就是基于 Lie-group transport 的因果 affine recurrence。当前路线聚焦于三件事：

1. 把这个 affine operator 的数学定义、数值稳定性和实现路径讲清楚。
2. 先用可信的 reference implementation 建立正确性和 benchmark 基线。
3. 再向 Triton / CUDA kernel、长上下文训练和语言建模评测推进。

## 当前状态

- 当前最接近论文主线的标准化结果见 [paper/results_snapshot.md](./paper/results_snapshot.md)。
  其中，coupled write-forget MOGT 在 single-slot train512 -> eval8192
  任务上三种子达到 `100.00% +/- 0.00%`，当前标准 NoPE Transformer 为
  `78.65% +/- 12.63%`；在 tracked 4-slot final-query curriculum 任务上，
  slot-addressed MOGT 于 4096 context 达到 `94.27% +/- 2.39%`，NoPE 为
  `21.35% +/- 6.31%`，参数接近的 HF-Mamba d192 为
  `19.79% +/- 10.97%`。
- 当前 core-operator 系统探针见
  [benchmark_runs/throughput_core_operator_summary_20260504.md](./benchmark_runs/throughput_core_operator_summary_20260504.md)：
  在 `d_model=768`、rank 16、batch size 1 的 L4 测量中，affine
  `triton_hybrid` scan core 在 16k/32k 比 attention core 更快；这是
  core-only 结果，不是端到端吞吐结论。
- 当前 backbone-level 前向探针见
  [benchmark_runs/backbone_throughput_summary_20260504.md](./benchmark_runs/backbone_throughput_summary_20260504.md)：
  同样在 L4 上，2-layer `d_model=768` 的 identity coupled MOGT backbone
  在 8k/16k 仍不快于 NoPE Transformer，但 32k 前向为 `77.56ms` vs
  `97.05ms`。这仍不包含 LM head、loss、反传或生成式 KV-cache。
- 已新增第一版 hybrid MOGT/Transformer 语言模型骨架
  [model_hybrid.py](/home/lkc/MOGT/model_hybrid.py:1) 和预算训练入口
  [train_budget_hybrid.py](/home/lkc/MOGT/train_budget_hybrid.py:1)。一个
  `ctx=8192, d_model=128, 2 layers, 10 steps, seed=42` 的 wiring pilot 中，
  alternating hybrid 的 best val loss 为 `10.4828`，同规格 Transformer 为
  `10.5214`。这是可训性信号，不是语言建模胜利声明。
- 第一版 4-layer hybrid ratio pilot 见
  [benchmark_runs/hybrid_lm_ratio_sweep_summary_20260504.md](./benchmark_runs/hybrid_lm_ratio_sweep_summary_20260504.md)：
  在 `ctx=8192, d_model=128, 5 steps, seed=42` 下，0/25/50/75/100% MOGT
  层的 val loss 分别为 `10.6355 / 10.6159 / 10.6100 / 10.5838 / 10.5526`。
  这只是 1-batch validation 的早期比例扫描。加入
  `--zero-init-attention-out` 后，0% attention-only 改善到 `10.6032`，
  而 25/50/75/100% MOGT 为 `10.5540 / 10.5543 / 10.5532 / 10.5526`；
  因此原始单调趋势部分来自初始化差异。
- 在同一初始化控制下把 0/25/100% 三个点延长到 `50 steps` 后，val loss
  为 `9.5796 / 9.5618 / 9.6249`。这更支持“低比例 MOGT 插层”而不是
  “纯 MOGT 立即替代 attention”。
- 随后三种子 `50 steps` 小扫把 0/25/50% 的 val loss 均值定为
  `9.5865 / 9.5784 / 9.6034`，标准差 `0.0091 / 0.0163 / 0.0167`。
  25% 只有弱正信号，不能当成显著 LM 优势。
- 三种子 `200 steps` 加宽验证后，0% vs 25% MOGT 插层的 val loss 均值为
  `7.6069 / 7.4897`，标准差 `0.0091 / 0.0107`。这是目前最明确的
  hybrid LM 小模型正信号，但仍只是 WikiText-103、`ctx=8192`、4-layer
  小预算实验。
- 同预算下的 single-MOGT-layer 位置消融见
  [benchmark_runs/hybrid_layer_position_summary_20260504.md](./benchmark_runs/hybrid_layer_position_summary_20260504.md)：
  attention-only 为 `7.6069 +/- 0.0091`；MOGT 放在 layer1/2/3 的三种子
  val loss 为 `7.4897 +/- 0.0107`、`7.4675 +/- 0.0110`、
  `7.4539 +/- 0.0092`。layer0 目前只有 seed42，val loss 为 `7.5189`。
  这提示后层插入可能更适合小型 LM，但还需要更长训练和更大模型复核。
- 第一轮 late-layer scale-up 已完成：
  [benchmark_runs/hybrid_layer_position_steps500_summary_20260504.md](./benchmark_runs/hybrid_layer_position_steps500_summary_20260504.md)
  在 `500 steps`、8 个 validation batches、三种子下，attention-only
  val loss 为 `6.8137 +/- 0.0139`，layer3-MOGT 为
  `6.7241 +/- 0.0132`。这说明 200-step 的 hybrid 正信号没有被更长训练
  立即抹平，但仍需更大模型、更长 token budget 和标准 LM 数据集验证。
- `1000 steps` 续验也已完成：
  [benchmark_runs/hybrid_layer_position_steps1000_summary_20260504.md](./benchmark_runs/hybrid_layer_position_steps1000_summary_20260504.md)
  在同一设置下，attention-only 为 `6.3868 +/- 0.0098`，layer3-MOGT 为
  `6.3357 +/- 0.0128`。优势缩小但仍跨三种子存在，下一步应该检查更大
  `d_model/layers` 或更长 token budget 下是否继续保持。
- 第一轮宽度放大也完成：
  [benchmark_runs/hybrid_scale_d192_l4_steps500_summary_20260504.md](./benchmark_runs/hybrid_scale_d192_l4_steps500_summary_20260504.md)
  在 `d_model=192, 4 layers, 500 steps` 下，attention-only 为
  `6.5024 +/- 0.0141`，layer3-MOGT 为 `6.4544 +/- 0.0134`。这说明
  hybrid 正信号已经从 `d_model=128` 迁移到更宽模型，但仍只是小型
  WikiText-103 预算。
- 同一宽度推到 `1000 steps` 后出现重要边界：
  [benchmark_runs/hybrid_scale_d192_l4_steps1000_summary_20260504.md](./benchmark_runs/hybrid_scale_d192_l4_steps1000_summary_20260504.md)
  中，attention-only 为 `6.0997 +/- 0.0126`，layer2-MOGT 为
  `6.1049 +/- 0.0147`，layer3-MOGT 为 `6.1337 +/- 0.0162`，双中层
  layers1+2 MOGT 为 `6.1416 +/- 0.0180`。这意味着当前 layer3 单插层优势
  不能被表述为随训练预算单调保持；layer2 近似追平 attention，但还没有胜出，
  简单增加 MOGT 层数也没有解决。下一步需要找更合适的训练配方。
- 学习率公平诊断见
  [benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_summary_20260504.md](./benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_summary_20260504.md)：
  `lr=5e-4` 同时强化了 attention 和 layer2-MOGT，但 attention-only 仍更好，
  为 `5.8753 +/- 0.0108`，layer2-MOGT 为 `5.9058 +/- 0.0430`。因此
  当前 LM 路线必须诚实表述为“早期/中等预算有正信号，但在更公平调参后
  尚未超过 Transformer”。
- 新增 residual-scale sweep 诊断见
  [benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_mogtscale_sweep_seed42_20260505.md](./benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_mogtscale_sweep_seed42_20260505.md)：
  在同一 `d_model=192, layer2, lr=5e-4, seed=42` 设置下，MOGT readout
  residual scale `0.25/0.5/0.75/1.0` 的 val loss 分别为
  `5.9199 / 5.9110 / 5.9174 / 5.9545`；同 seed attention-only 为
  `5.8877`。这说明融合强度是有效杠杆，最佳固定点暂在 `0.5` 附近，
  但还不是 LM 胜利声明。
- scale `0.5` 的三种子复核见
  [benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_mogtscale0p5_summary_20260505.md](./benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_mogtscale0p5_summary_20260505.md)：
  attention-only 为 `5.8753 +/- 0.0108`，layer2-MOGT scale `0.5` 为
  `5.8775 +/- 0.0293`。它显著好于 unscaled layer2-MOGT
  (`5.9058 +/- 0.0430`)，并在 seeds `7/123` 上超过同 seed attention，
  但三种子均值仍高 `0.0022`。当前结论是 residual scaling 修复了大部分
  融合问题，尚未构成 Transformer 胜利。
- learned residual mixing 已接入最小版本：`--mogt-residual-gate` 会给每个
  MOGT block 增加一个可学习标量门，`--mogt-residual-gate-init` 控制初始
  sigmoid 值。smoke 报告
  [benchmark_runs/hybrid_residual_gate_smoke.json](./benchmark_runs/hybrid_residual_gate_smoke.json)
  已通过；但 seed42 预算诊断
  [benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_mogtgate0p5_diagnostic_seed42_20260505.md](./benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_mogtgate0p5_diagnostic_seed42_20260505.md)
  是负结果：固定 scale `0.5` 为 `5.9110`，naive learned gate init `0.5`
  为 `5.9752`。下一步若继续门控，需要更低门控学习率、正则、延迟解冻或
  有界 schedule。
- residual-scale schedule 也已接入并完成第一轮 seed42 诊断：
  `--mogt-residual-scale-start` 与 `--mogt-residual-scale-warmup-steps` 会将
  MOGT readout residual scale 线性升到 `--mogt-residual-scale`。但
  [benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_mogtsched0p25to0p5s250_diagnostic_seed42_20260505.md](./benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_mogtsched0p25to0p5s250_diagnostic_seed42_20260505.md)
  是负结果：`0.25 -> 0.5` over `250 steps` 得到 `5.9497`，差于 fixed
  scale `0.5` 的 `5.9110`。简单 scale warmup 不值得立即扩展成三种子。
- 更有价值的新信号来自 optimizer partition：`--mogt-lr-mult` 已接入，
  只调整 MOGT block 参数学习率倍率。在同一 seed42/d192/layer2/lr=5e-4
  设置中，fixed residual scale `0.5` 加 `--mogt-lr-mult 0.5` 达到
  `5.8601`，反超 same-seed attention `5.8877`，详见
  [benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_mogtscale0p5_mogtlr0p5_diagnostic_seed42_20260505.md](./benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_mogtscale0p5_mogtlr0p5_diagnostic_seed42_20260505.md)。
  三种子复核见
  [benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_mogtscale0p5_mogtlr0p5_summary_20260505.md](./benchmark_runs/hybrid_scale_d192_l4_steps1000_lr5e4_mogtscale0p5_mogtlr0p5_summary_20260505.md)：
  attention-only 为 `5.8753 +/- 0.0108`，layer2-MOGT 为
  `5.8511 +/- 0.0117`，三种子全赢，均值优势 `-0.0242`。这是当前最强的
  hybrid LM 证据，但仍属于小模型/短预算 WikiText-103 结果。
- 第一轮宽度迁移边界见
  [benchmark_runs/hybrid_scale_d256_l4_steps1000_lr5e4_mogtscale0p5_mogtlr0p5_summary_seed42_20260505.md](./benchmark_runs/hybrid_scale_d256_l4_steps1000_lr5e4_mogtscale0p5_mogtlr0p5_summary_seed42_20260505.md)：
  在 `d_model=256`、seed42、同预算下，attention-only 为 `5.7550`，
  layer2-MOGT 为 `5.7558`。这基本打平但未赢，说明 d192 配方没有随宽度
  崩掉，但也还没有形成宽度 scaling claim。
- 这些结果支持“受控状态跟踪机制论文”，不支持“已经全面取代
  Transformer 的语言模型论文”。
- 训练主路径现在使用真实的 affine transport 递推，代码在 [model_mogt.py](/home/lkc/MOGT/model_mogt.py:71) 和 [affine_scan.py](/home/lkc/MOGT/affine_scan.py:14)。
- `sequential` 是当前可信的 reference path，用于训练和后续验证。
- `parallel_reference` 是 Hillis-Steele 风格的并行参考实现，用于验证 affine operator 的结合律，不是最终高性能实现。
- `block_reference` 是更接近未来 Triton kernel 的 block-local scan + carry reference，用来先验证块级执行形态。
- `triton_hybrid` 是当前最新的过渡版：前向的块内 local scan 走 Triton，块间 carry 先用 reference 组合；反向则使用基于 affine recurrence 的 custom reverse scan 重算，以保证梯度能传回 `phi_conn / phi_val`。
- 在长上下文训练上，当前主线已经在单张 L4 24GB 上跑通 `12 layers x d_model 768 x ctx 32768` 的真实短程训练，并固化了一个 `200-step` 的 `baseline_v1` 候选；要站住这一级别，需要同时启用 `expandable_segments` allocator、chunked LM-head/loss 和 activation checkpointing。
- `ctx=32768` 的最新 profile 见 [benchmark_runs/training_profile_context32768_20260423.json](/home/lkc/MOGT/benchmark_runs/training_profile_context32768_20260423.json:1)。在这一级别，把 block carry 从 `sequential` 切到 `doubling` 后，`carry_scan` 约从 `947.9ms` 降到 `54.9ms`；当前更值得继续优化的热点已经变成 `matrix_exp`、`carry_apply` 和 chunked `lm_head_loss`。
- 我们已经补上了一个实验性的 `connection_impl="cayley"` 路径；在 `32k` 的单步 profile 上，它把总步时从约 `3337ms` 进一步压到 `2479ms`，详见 [benchmark_runs/training_profile_connection_impl_ctx32768_20260423.json](/home/lkc/MOGT/benchmark_runs/training_profile_connection_impl_ctx32768_20260423.json:1)。在 `50-step + 5-batch val` 对照里，`cayley` 同时取得更低验证 loss 和约 `1.41x` 端到端加速，详见 [benchmark_runs/connection_impl_eval50_ctx32768_20260428.json](/home/lkc/MOGT/benchmark_runs/connection_impl_eval50_ctx32768_20260428.json:1)。`baseline_v1` 已完成 seed `42/7/123` 的 `200-step` 初步复核，三 seed best val loss 均值为 `6.4066`，PPL 均值为 `607.45`，详见 [benchmark_runs/baseline_v1_cayley_ctx32768_multiseed_20260428.json](/home/lkc/MOGT/benchmark_runs/baseline_v1_cayley_ctx32768_multiseed_20260428.json:1)。这条线仍是 opt-in，因为它是近似指数映射，而且当前 `ctx=32768` 验证实际只有 `5` 个 batches，还需要更宽验证协议复核。
- 第一轮 checkpoint-only 跨 context 复核已经完成：seed `42/7/123` 在 `ctx=8192/16384/32768` 上的 loss 均值分别为 `6.4065 / 6.4186 / 6.4067`，对应 PPL 均值为 `607.33 / 614.73 / 607.45`，详见 [benchmark_runs/baseline_v1_cayley_multiseed_eval_ctx8192_16384_32768.json](/home/lkc/MOGT/benchmark_runs/baseline_v1_cayley_multiseed_eval_ctx8192_16384_32768.json:1)。这加强了“短程训练稳定”的证据，但还不等价于完整语言建模 benchmark。
- 已补第一轮外部预训练锚点：GPT-2 Small 在 GPT-2 token stream 的 `ctx=1023` 上 PPL 为 `25.25`；Mamba-130M HF 用原生 tokenizer 在 `ctx=8192/16384/32768` 上 PPL 为 `20.05 / 24.60 / 59.07`，详见 [benchmark_runs/external_pretrained_baseline_table_20260429.md](/home/lkc/MOGT/benchmark_runs/external_pretrained_baseline_table_20260429.md:1)。这些是尺度锚点，不是训练预算匹配的公平对照。
- 第一轮训练预算匹配 scratch baseline 已完成三 seed：同 WikiText-103/GPT-2 token stream、同 `ctx=32768`、同 `200` optimizer steps 下，Scratch Mamba SSM best val loss 均值为 `9.6168`，PPL 均值为 `15059.19`，明显弱于 MOGT `baseline_v1` 的 `6.4067` / `607.45`。结果和恢复说明见 [benchmark_runs/budget_matched_scratch_baseline_table_20260429.md](/home/lkc/MOGT/benchmark_runs/budget_matched_scratch_baseline_table_20260429.md:1)。这仍只是 200-step 负对照，不是完整语言建模结论。
- `transport_triton` 仍然保留，但它只覆盖 `U` 的 transport proxy，不等价于完整的 affine recurrence，不能再被当成完整模型吞吐结论。
- `benchmark_perplexity.py`、`benchmark_passkey.py`、`benchmark_lifelong.py`、`benchmark_scaling.py` 目前仍保留，但属于探索性脚本，不应在没有额外校准前作为强结论来源。

## 核心算子

MOGT block 的主干可拆成两部分：

1. 从输入生成局部群元素 `U_t` 和局部值张量 `V_t`
2. 在序列维度上执行 affine scan

其组合律为：

`(U_next, V_next) ⊗ (U_prev, V_prev) = (U_next @ U_prev, U_next @ V_prev + V_next)`

这条组合律是整个项目当前最重要的工程接口。它决定了：

- reference implementation 如何写
- parallel scan 是否数学正确
- Triton kernel 应该以什么状态布局和 carry 形式实现

## 快速开始

### 1. 环境

这个仓库默认假设你已经有可用的 PyTorch 环境。若你在 GCP L4 上开发，建议先确认 CUDA 版 PyTorch 已安装，再安装项目依赖：

```bash
pip install -r requirements.txt
```

### 2. 基础连通性

```bash
python3 model_mogt.py
python3 sanity_affine_scan.py
```

`sanity_affine_scan.py` 会验证 `sequential` 和 `parallel_reference` 两条 affine scan 路径是否逐元素对齐。
当前它也会同时校验 `block_reference`。

```bash
python3 sanity_triton_gradients.py
python3 sanity_triton_training.py
```

这两个脚本分别检查：

- `triton_hybrid` 是否真的把梯度传回关键投影层
- `triton_hybrid` 和 `sequential` 在一个小型训练步上的 logits / loss / 梯度 / 参数更新是否保持对齐

### 3. 训练

```bash
python3 train.py
```

当前训练默认使用 `sequential` affine reference path。它的价值主要是：

- 验证模型在真实递推下是否稳定训练
- 给未来 Triton kernel 留下可比较的 baseline checkpoint

如果你要在 L4 上直接推进长上下文训练，可以显式切到 `triton_hybrid`：

```bash
MOGT_SCAN_IMPL=triton_hybrid python3 train.py
```

`train.py` 现在会在 `ctx >= 32768` 时自动启用：

- `PYTORCH_ALLOC_CONF=expandable_segments:True`
- chunked LM-head/loss
- activation checkpointing
- 对 `triton_hybrid` 自动切到 `doubling` carry scan

这样做的目的不是追求“默认最优”，而是让 `32k` 这一级别先稳定可训。

如果你要试验更便宜的 connection map，可以额外指定：

```bash
MOGT_SCAN_IMPL=triton_hybrid MOGT_CONNECTION_IMPL=cayley python3 train.py
```

当前这条 `cayley` 路径已经在 `32k` 的 `50-step + 5-batch val` 对照中胜过 `matrix_exp`，并在 `baseline_v1` 的三 seed `200-step` run 中保持稳定下降；但还没有被提升为默认设置，下一步应把同一批 checkpoint 放到更宽验证协议里复核。

如果要直接跑当前固化的 L4 候选基线，可以使用：

```bash
MOGT_RUN_PRESET=baseline_v1 python3 train.py
```

这个 preset 默认使用 `ctx=32768, d_model=768, layers=12, rank=16, triton_hybrid, cayley, 200 steps`，验证上限为 `10` 个 batch；在当前 WikiText-103 `ctx=32768` 验证切分下实际会跑满 `5` 个 batch。它会把 checkpoint 写到独立的 `baseline_v1` 目录。若只想检查配置是否能连通，可以先跑：

```bash
MOGT_RUN_PRESET=baseline_v1_smoke python3 train.py
```

如果你要顺手带上一个轻量验证集信号，可以再加：

```bash
MOGT_EVAL_MAX_BATCHES=2 MOGT_EVAL_AT_END=1 MOGT_SAVE_BEST=1 python3 train.py
```

这会在训练结束时输出 `Val loss / PPL`，并把当前最优验证结果写到 `mogt_best.pt`。最近一轮正式 `baseline_v1` 单 seed 结果见 [benchmark_runs/baseline_v1_cayley_ctx32768_seed42_20260428.json](/home/lkc/MOGT/benchmark_runs/baseline_v1_cayley_ctx32768_seed42_20260428.json:1)，三 seed 汇总见 [benchmark_runs/baseline_v1_cayley_ctx32768_multiseed_20260428.json](/home/lkc/MOGT/benchmark_runs/baseline_v1_cayley_ctx32768_multiseed_20260428.json:1)。

如果要对已有 checkpoint 跑更宽验证协议，可以使用 checkpoint-only eval 入口：

```bash
python3 evaluate_checkpoints.py \
  --checkpoint mogt_checkpoints/baseline_v1_cayley_ctx32768_seed42 \
  --context-lengths 8192 16384 32768 \
  --max-batches 20 \
  --output benchmark_runs/baseline_v1_seed42_eval_ctx8192_16384_32768.json
```

这个脚本默认会读取目录下的 `mogt_best.pt`，并沿用 `triton_hybrid + cayley` 的候选配置。

第一轮三 seed 结果已经写入 [benchmark_runs/baseline_v1_cayley_multiseed_eval_ctx8192_16384_32768.json](/home/lkc/MOGT/benchmark_runs/baseline_v1_cayley_multiseed_eval_ctx8192_16384_32768.json:1)，表格草稿见 [benchmark_runs/baseline_v1_cayley_eval_table_20260429.md](/home/lkc/MOGT/benchmark_runs/baseline_v1_cayley_eval_table_20260429.md:1)。在 `ctx=8192` 下每个 checkpoint 跑满 `20` 个 validation batches，在 `ctx=16384` 下实际有 `13` 个 batches，在 `ctx=32768` 下仍是 `5` 个 batches。

### 4. 核心 benchmark

```bash
python3 benchmark_throughput.py --device cuda --lengths 1024 2048 4096 8192 16384
```

这个脚本现在会明确区分：

- `Affine Scan (Sequential Ref)`
- `Affine Scan (Parallel Ref)`
- `Affine Scan (Block Ref)`
- `Affine Scan (Triton Hybrid)`
- `Transport-only Triton (Legacy Proxy)`
- `Attention Core (Flash/SDPA)`

### 5. 外部预训练锚点

```bash
python3 evaluate_hf_baselines.py \
  --models state-spaces/mamba-130m-hf \
  --context-lengths 8192 16384 32768 \
  --tokenization-mode native_text \
  --max-batches 20 \
  --output benchmark_runs/hf_mamba_130m_native_eval_ctx8192_16384_32768.json
```

`native_text` 使用模型自己的 tokenizer，因此适合作为预训练尺度锚点；若要和 MOGT 共享 GPT-2 token stream，则只能使用 tokenizer/vocab 兼容的模型，例如：

```bash
python3 evaluate_hf_baselines.py \
  --models gpt2 \
  --context-lengths 1023 \
  --tokenization-mode gpt2_stream \
  --max-batches 20 \
  --output benchmark_runs/hf_gpt2_stream_eval_ctx1023.json
```

### 6. 训练预算匹配 baseline

```bash
python3 train_budget_baseline.py \
  --run-name mamba_scratch_budget_v1_ctx32768_seed42 \
  --context-length 32768 \
  --d-model 768 \
  --num-layers 24 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --max-steps 200 \
  --eval-interval 50 \
  --eval-max-batches 10 \
  --seed 42 \
  --report-output benchmark_runs/mamba_scratch_budget_v1_ctx32768_seed42.json \
  --checkpoint-dir baseline_checkpoints/mamba_scratch_budget_v1_ctx32768_seed42
```

这个入口训练一个从零初始化的 Mamba-style baseline，复用 MOGT 的 GPT-2 tokenizer 和 WikiText-103 dataloader。第一轮三 seed 结果见 [benchmark_runs/budget_matched_scratch_baseline_table_20260429.md](/home/lkc/MOGT/benchmark_runs/budget_matched_scratch_baseline_table_20260429.md:1)。

## 代码结构

- `affine_scan.py`: affine recurrence 的公共定义与 reference implementation。
- `affine_scan.py` 现在同时包含 `sequential`、`parallel_reference` 和 `block_reference` 三层参考路径。
- `triton_scan.py`: `transport_triton` 旧 proxy 与新的 `triton_hybrid` 过渡版。
- `model_mogt.py`: MOGT block 与语言模型封装，当前接入真实 affine scan。
- `benchmark_throughput.py`: 新的 operator-first benchmark，显式区分真实 affine core 与 legacy transport proxy。
- `sanity_affine_scan.py`: affine scan 对齐测试。
- `train.py`: 训练入口，支持环境变量配置与 `MOGT_RUN_PRESET=baseline_v1`。
- `evaluate_checkpoints.py`: checkpoint-only 验证入口，用于把同一 checkpoint 放到多个 context length 下评估。
- `evaluate_hf_baselines.py`: HuggingFace CausalLM 预训练锚点评测入口，显式区分 GPT-2 token stream 和模型原生 tokenizer。
- `train_budget_baseline.py`: 训练预算匹配 scratch baseline 入口，当前支持 Mamba SSM，带 chunked loss、activation checkpointing、resume 和周期 checkpoint。
- `model_hybrid.py` / `train_budget_hybrid.py`: hybrid MOGT/Transformer
  原型与预算训练入口，用于 Phase 5 的 layer-ratio LM 实验。
- `run_hybrid_lm_sweep.py` / `summarize_hybrid_lm_sweep.py`: hybrid
  layer-ratio pilot 的批量运行与汇总脚本。
- `dataset.py`: WikiText-103 数据管线。
- `optimizer_mogt.py`: 持续学习方向的实验性优化器。

## Benchmark 分层

### Tier 1: 当前可信主线

- `sanity_affine_scan.py`
- `benchmark_throughput.py`
- `train.py` 上的稳定性与验证 loss 追踪

这些脚本直接服务于当前主问题：真实 affine transport operator 能不能正确、稳定、可扩展地跑起来。

### Tier 2: 探索性脚本

- `benchmark_perplexity.py`
- `benchmark_passkey.py`
- `benchmark_lifelong.py`
- `benchmark_scaling.py`

这些脚本暂时保留，但不再代表当前版本的主 deliverable。后续只有在以下条件满足后才会重新上升为主线：

1. affine scan forward kernel 成型
2. 训练路径切换到新 kernel 后保持数值一致
3. baseline 和评测协议被重新校准

## 开发路线

项目当前采用 operator-first 路线：

1. 固化 affine operator 的 reference semantics
2. 实现 Triton affine forward kernel
3. 做 kernel/reference 的数值一致性测试
4. 用 custom reverse-scan backward 把 kernel 安全接回训练主路径
5. 用 chunked loss + activation checkpointing 把长上下文训练稳定接上
6. 再重启大规模训练与语言建模评测

## 目前不再宣称的内容

为了让仓库叙事和实现一致，当前版本不再把以下内容当作已证明结论：

- “完整 MOGT 已实现并行 O(N) 训练主路径”
- “现有 throughput 图已经代表完整模型路径”
- “lifelong / passkey / scaling 已经构成稳定论文结论”
- “当前仓库已经是零配置可复现实验包”

这些方向仍然重要，但会在核心算子和 benchmark 站稳之后逐步回归。

## 推荐使用方式

如果你现在在 GCP L4 上继续推进，最值得优先投入时间的是：

1. 把 checkpoint-only eval 和 scratch baseline 表格生成自动化，后续每个 baseline 都输出同一 JSON schema
2. 补一个同 tokenizer / 同 token budget 的 scratch Transformer baseline，避免只和 Mamba-family 对照
3. 继续压缩 `carry_apply` 与 chunked `lm_head_loss` 的热点

等这三件事站稳，再回头拉更长上下文训练、PPL、passkey 和持续学习，会更省钱也更可信。
