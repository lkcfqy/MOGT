# MOGT 路线重整：GCP L4 上的 Operator-First MVP

这份清单不再把项目当作“已经完成的顶会级全套结论”，而是把它收敛成一个更可信、更适合在 GCP L4 上推进的 MVP：

先证明核心 affine transport operator，再逐步恢复训练、评测和更大的理论叙事。

---

## 核心目标

当前版本的唯一主目标：

**把 MOGT 的真实递推主干**

`H_t = U_t @ H_{t-1} + V_t`

**做成一个正确、稳定、可 benchmark、可替换为 Triton kernel 的 operator。**

---

## 为什么要重整路线

原路线的问题不在于方向完全错误，而在于并行推进了太多层：

- 大模型训练
- throughput 论文图
- PPL 对比
- passkey
- lifelong
- scaling law

这些工作同时推进时，最容易出现的情况就是：

1. benchmark 测到的是 proxy，不是主路径
2. 文档叙事跑在实现前面
3. GPU 时间被大训练消耗，但核心算子接口还没彻底定型

所以现在的策略是先收缩，再重新展开。

---

## 新的 MVP 分期

## Phase A: 算子定义与 reference 路径

**目标**：把 affine operator 的语义固化。

- [x] 抽取公共 affine recurrence 模块 `affine_scan.py`
- [x] 提供 `sequential` reference implementation
- [x] 提供 `parallel_reference` implementation
- [x] 提供 `block_reference` implementation，模拟 block-local scan + carry
- [x] 在 `model_mogt.py` 中接入 `scan_impl`
- [x] 保留 `transport_triton` 作为 legacy proxy，但从主叙事中剥离
- [x] 加入 `sanity_affine_scan.py` 做 reference 对齐验证

**完成标准**

- `sequential` 与 `parallel_reference` 前缀结果逐元素对齐
- 模型前向可以在两条实现上保持形状与数值接近

---

## Phase B: 真实 operator benchmark

**目标**：只测真实 affine core，不再混淆 transport proxy。

- [x] 重写 `benchmark_throughput.py`
- [x] 显式输出四类曲线：
  - `Affine Scan (Sequential Ref)`
  - `Affine Scan (Parallel Ref)`
  - `Affine Scan (Block Ref)`
  - `Affine Scan (Triton Hybrid)`
  - `Transport-only Triton (Legacy Proxy)`
  - `Attention Core (Flash/SDPA)`
- [x] 增加 CPU/CUDA 分支兼容
- [x] 在 L4 上跑一轮正式 CUDA benchmark
- [x] 记录不同长度下的 wall-clock 与 OOM 边界

**完成标准**

- throughput 图不再把 legacy transport proxy 冒充完整模型路径
- L4 上至少覆盖 `1k -> 16k` 的真实 affine core 测试

---

## Phase C: Triton affine forward kernel

**目标**：把 reference semantics 换成真正可用的高性能 forward kernel。

- [x] 设计 block-local affine scan 的状态布局
- [x] 明确跨 block carry 的 `(U_carry, V_carry)` 接口
- [x] 实现第一版 Triton hybrid prototype
- [x] 跑 kernel/reference 数值一致性测试
- [x] 接回 `model_mogt.py` 作为实验性 `scan_impl="triton_hybrid"`
- [x] 用 custom reverse-scan backward 打通训练期梯度
- [ ] 将 hybrid prototype 推进为更完整的纯 Triton affine forward

**完成标准**

- kernel 输出与 reference 在容许误差内一致
- 在 L4 上对比 `sequential`、`block_reference` 和 `triton_hybrid` 有实测速度收益
- 在长上下文 profile 里明确下一阶段主瓶颈，而不是继续把所有时间花在“先跑更长”

---

## Phase D: 训练回归

**目标**：只在 operator 稳定后重新烧训练。

- [x] 重新启动短程训练，先验证 loss 是否稳定
- [x] 做 kernel/reference 的小型单步训练对照
- [x] 在 L4 上将真实短程训练推进到 `12 layers x d_model 768 x ctx 32768`
- [x] 在训练路径中接入 chunked LM-head/loss、activation checkpointing 和 allocator 配置
- [x] 给 connection map 增加可切换后端，并在 `ctx=32768` 上完成 `matrix_exp vs cayley` 的 profile / smoke 对比
- [x] 增加最基础的验证集评估与 checkpoint 管理
- [x] 增加 checkpoint-only eval 入口 `evaluate_checkpoints.py`，用于跨 context length 复核已有 checkpoint
- [x] 跑完 `baseline_v1` seed `42/7/123` 的 checkpoint-only 跨 context 复核：`ctx=8192/16384/32768` loss 均值为 `6.4065 / 6.4186 / 6.4067`
- [x] 输出第一版 eval 表格草稿 `benchmark_runs/baseline_v1_cayley_eval_table_20260429.md`
- [x] 完成 `50-step + 5-batch val` 的 `matrix_exp vs cayley` 对照，`cayley` 暂时成为 `baseline_v1` 的最强候选
- [x] 固化一个 `baseline_v1` run 配置：`MOGT_RUN_PRESET=baseline_v1 python3 train.py`
- [x] 跑完 `baseline_v1` seed 42：`200-step` 后 best val loss `6.3407`，PPL `567.22`
- [x] 整理 `baseline_v1` seed `42/7/123` 的 `200-step` 初步复核：三 seed best val loss 均值 `6.4066`，PPL 均值 `607.45`

**完成标准**

- 新 kernel 不引入明显发散
- 小型训练回归中，`triton_hybrid` 与 `sequential` 的 logits / loss / 梯度 / 参数更新保持对齐
- 恢复训练后的 loss 曲线与 reference path 同量级
- 长上下文训练不再依赖手动记忆一串易错的环境变量
- 对可选近似路径，已经拿到多 seed 初步信号；但还需要更多验证 batch 或跨 context 验证，才能考虑升级默认值

---

## Phase E: 评测回归

**目标**：在训练主路径稳定后，重新启用探索脚本。

- [ ] 重新审查 `benchmark_perplexity.py`
- [x] 增加 HuggingFace 预训练锚点评测入口 `evaluate_hf_baselines.py`
- [x] 跑完第一轮 GPT-2 / Mamba-130M 预训练锚点，并明确标注它们不是训练预算匹配公平对照
- [x] 增加训练预算匹配 scratch baseline 入口 `train_budget_baseline.py`
- [x] 跑完 Scratch Mamba SSM seed `42/7/123`：同 WikiText-103/GPT-2 stream、`ctx=32768`、`200-step` 下 best val loss 均值 `9.6168`，PPL 均值 `15059.19`
- [ ] 重新定义 `benchmark_passkey.py` 为“合成长程记忆实验”
- [ ] 重新审查 `benchmark_lifelong.py` 的任务构造
- [ ] 暂缓 `benchmark_scaling.py`，直到训练协议可信

**完成标准**

- 所有评测脚本都明确写出它们的假设和边界
- 不再把探索性图表包装成最终结论

---

## 当前优先级

在 GCP L4 上，接下来最值得投入 GPU 时间的顺序是：

1. 把 checkpoint-only eval 和 scratch baseline 表格生成自动化，后续每个 baseline 都输出同一格式
2. 补一个同 tokenizer / 同 token budget 的 Scratch Transformer baseline，避免只和 Mamba-family 对照
3. 继续压缩长上下文训练的一步 wall-clock
4. 优化 `carry_apply` 与 chunked `lm_head_loss`
5. 再决定是否继续推 `65536`

当前不优先：

- 盲目继续拉更长长度
- 长时间大训练
- scaling law
- 持续学习结论
- 对外宣称论文级结果

---

## 暂停项

以下内容先降级为“后续方向”，不作为当前 MVP 成功标准：

- “完整 MOGT 已经并行化”
- “现有 benchmark 已证明顶会级结论”
- “PPL / passkey / lifelong / scaling 全线收敛”
- “项目已经零配置复现”

这些并不是被放弃，而是需要建立在 operator-first 路线成功之后。

---

## 里程碑定义

### MVP 完成的标志

- affine operator 定义稳定
- reference 与 Triton forward 对齐
- L4 上有可信的 operator benchmark
- 模型可在新 kernel 下恢复稳定训练

### 之后才能进入的阶段

- 重启中长程训练
- 重做语言建模和长上下文评测
- 重启更完整的论文叙事与图表
