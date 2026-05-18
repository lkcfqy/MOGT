# MOGT

MOGT 是一个长上下文状态跟踪研究仓库，核心关注“可扫描的矩阵值递归算子”能否在受控长上下文任务上承担显式写入、遗忘和按槽位路由。当前仓库是一次清理后的研究交接版本，重点服务论文复现、结果核对和新机器启动。

## 当前结论

当前最稳妥的结论是：耦合 write-forget gating 与 prefix-conditioned slot addressing 可以让 scan-compatible 的矩阵值 recurrent operator 在若干受控长上下文状态跟踪任务上表现很好，而匹配的 NoPE Transformer 和 HF-Mamba baseline 在这些任务上更吃力。

这不是“替代 Transformer”的通用结论。语言模型混合层实验仍是边界探索，仓库里也保留了中性或负向结果。

## 结果快照

`paper/results_snapshot.md` 和 `docs/claim_ledger.md` 中记录了当前主张依据。已整理结果包括：

- 单槽 overwrite：Coupled MOGT 在 512 到 8192 长度上保持 `100%`，NoPE Transformer 在 8192 长度上约为 `78.65% ± 12.63`。
- 4 槽 final-query routing：slot-addressed MOGT 在 4096 长度上约为 `94.27% ± 2.39`，NoPE Transformer 约为 `21.35% ± 6.31`，HF-Mamba d192 约为 `19.79% ± 10.97`。
- Core timing：32768 长度下 affine Triton core 约 `5.48 ms`，attention core 约 `27.32 ms`，注意这是 core-only 对比。
- Backbone forward：d768、4 层、32768 长度下 MOGT 约 `77.56 ms`，Transformer 约 `97.05 ms`；在 8K/16K 长度上 Transformer 仍有竞争力。

## 主要目录

- `docs/PROJECT_FREEZE_20260505.md`：当前冻结状态说明。
- `docs/NEW_VM_QUICKSTART.md`：新机器启动步骤。
- `docs/claim_ledger.md`：论文主张与证据台账。
- `paper/`：论文草稿和结果快照。
- `src/`：MOGT、hybrid 模型、affine scan、Triton scan 和训练入口。
- `reports/`：实验报告与汇总数据。
- `scripts/`：环境启动、结果汇总和校验脚本。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
python scripts/summarize_standard_reports.py
python scripts/summarize_paper_results.py
python scripts/validate_experiment_reports.py
```

新机器可直接使用：

```bash
bash scripts/bootstrap_new_vm.sh
```

如需安装 FlashAttention、Mamba SSM 等 GPU 可选依赖，可在确认 CUDA/PyTorch 环境后再参考 `requirements-optional-gpu.txt`。

## 许可证

当前仓库未包含独立 `LICENSE` 文件。如需公开复用或分发，请先补充明确的开源许可证。
