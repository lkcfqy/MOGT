import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from chunked_lm_loss import chunked_linear_cross_entropy
from model_baseline_transformer import TransformerBlock, choose_num_heads
from model_mogt import MOGTBlock, RMSNorm


def _logit_from_probability(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


def _fractional_layer_indices(num_layers: int, fraction: float) -> set[int]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"mogt layer fraction must be in [0, 1], got {fraction}")
    count = max(0, min(num_layers, int(round(num_layers * fraction))))
    if count == 0:
        return set()
    if count == num_layers:
        return set(range(num_layers))

    centers = [(idx + 0.5) / num_layers for idx in range(num_layers)]
    chosen: set[int] = set()
    for slot_idx in range(count):
        target = (slot_idx + 0.5) / count
        available = [idx for idx in range(num_layers) if idx not in chosen]
        chosen.add(min(available, key=lambda idx: (abs(centers[idx] - target), idx)))
    return chosen


def _explicit_layer_indices(num_layers: int, indices: list[int] | tuple[int, ...]) -> set[int]:
    selected = set()
    for raw_idx in indices:
        idx = int(raw_idx)
        if idx < 0:
            idx = num_layers + idx
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"MOGT layer index {raw_idx} is out of range for {num_layers} layers")
        selected.add(idx)
    return selected


def hybrid_layer_indices(
    num_layers: int,
    pattern: str,
    mogt_layer_fraction: float | None = None,
    explicit_mogt_layer_indices: list[int] | tuple[int, ...] | None = None,
) -> set[int]:
    if explicit_mogt_layer_indices is not None:
        return _explicit_layer_indices(num_layers, explicit_mogt_layer_indices)
    if mogt_layer_fraction is not None:
        return _fractional_layer_indices(num_layers, mogt_layer_fraction)
    if pattern == "ratio_even":
        raise ValueError("pattern='ratio_even' requires mogt_layer_fraction")
    if pattern == "alternating":
        return {idx for idx in range(num_layers) if idx % 2 == 1}
    if pattern == "mogt_first_half":
        return set(range(num_layers // 2))
    if pattern == "mogt_second_half":
        return set(range(num_layers // 2, num_layers))
    if pattern == "all_mogt":
        return set(range(num_layers))
    if pattern == "all_transformer":
        return set()
    raise ValueError(f"unknown hybrid pattern: {pattern}")


class HybridBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        r: int = 16,
        rope_theta: float = 10000.0,
        pattern: str = "alternating",
        mogt_layer_fraction: float | None = None,
        explicit_mogt_layer_indices: list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mogt_layer_indices = hybrid_layer_indices(
            num_layers,
            pattern,
            mogt_layer_fraction=mogt_layer_fraction,
            explicit_mogt_layer_indices=explicit_mogt_layer_indices,
        )
        hidden_dim = int(8 * d_model / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        self.blocks = nn.ModuleList()
        self.block_types = []
        for layer_idx in range(num_layers):
            if layer_idx in self.mogt_layer_indices:
                self.blocks.append(MOGTBlock(d_model=d_model, r=r))
                self.block_types.append("mogt")
            else:
                self.blocks.append(
                    TransformerBlock(
                        d_model=d_model,
                        num_heads=num_heads,
                        hidden_dim=hidden_dim,
                        rope_theta=rope_theta,
                    )
                )
                self.block_types.append("transformer")
        self.norm_f = RMSNorm(d_model)
        self.gradient_checkpointing = False
        self.checkpoint_every_n = 1
        self.prefix_condition_position = None

    def configure_mogt(
        self,
        *,
        scan_impl: str = "triton_hybrid",
        connection_impl: str = "cayley",
        connection_damping: float = 0.999,
        scan_block_size: int = 256,
        mogt_residual_scale: float = 1.0,
        mogt_ffn_residual_scale: float = 1.0,
        mogt_residual_gate: bool = False,
        mogt_residual_gate_init: float = 0.5,
    ) -> None:
        for block_type, block in zip(self.block_types, self.blocks):
            if block_type != "mogt":
                continue
            block.scan_impl = scan_impl
            block.connection_impl = connection_impl
            block.connection_damping = connection_damping
            block.scan_block_size = scan_block_size
            block.mogt_residual_scale = mogt_residual_scale
            block.mogt_ffn_residual_scale = mogt_ffn_residual_scale
            block.mogt_residual_gate_enabled = bool(mogt_residual_gate)
            if mogt_residual_gate:
                init_logit = _logit_from_probability(mogt_residual_gate_init)
                gate_logit = getattr(block, "mogt_residual_gate_logit", None)
                if isinstance(gate_logit, nn.Parameter):
                    with torch.no_grad():
                        gate_logit.fill_(init_logit)
                else:
                    device = next(block.parameters()).device
                    block.mogt_residual_gate_logit = nn.Parameter(torch.tensor(init_logit, device=device))

    def set_mogt_residual_scale(self, scale: float) -> None:
        scale = float(scale)
        for block_type, block in zip(self.block_types, self.blocks):
            if block_type == "mogt":
                block.mogt_residual_scale = scale

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        prefix_condition = None
        if self.prefix_condition_position is not None:
            prefix_position = int(self.prefix_condition_position)
            if -x.size(1) <= prefix_position < x.size(1):
                prefix_condition = x[:, prefix_position, :]

        use_checkpoint = self.training and self.gradient_checkpointing
        checkpoint_every_n = max(1, int(self.checkpoint_every_n))
        for block_idx, (block_type, block) in enumerate(zip(self.block_types, self.blocks)):
            if use_checkpoint and (block_idx % checkpoint_every_n == 0):
                if block_type == "mogt":
                    x = checkpoint(block, x, prefix_condition, use_reentrant=False)
                else:
                    x = checkpoint(block, x, use_reentrant=False)
            else:
                if block_type == "mogt":
                    x = block(x, prefix_condition=prefix_condition)
                else:
                    x = block(x)
        return self.norm_f(x)


class HybridMOGTTransformerForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        r: int = 16,
        rope_theta: float = 10000.0,
        pattern: str = "alternating",
        mogt_layer_fraction: float | None = None,
        explicit_mogt_layer_indices: list[int] | tuple[int, ...] | None = None,
        zero_init_attention_out: bool = False,
    ):
        super().__init__()
        self.backbone = HybridBackbone(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            r=r,
            rope_theta=rope_theta,
            pattern=pattern,
            mogt_layer_fraction=mogt_layer_fraction,
            explicit_mogt_layer_indices=explicit_mogt_layer_indices,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.backbone.embedding.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if "phi_conn" in name:
                torch.nn.init.normal_(parameter, mean=0.0, std=1e-4)
            elif "theta_read" in name:
                torch.nn.init.zeros_(parameter)
            elif zero_init_attention_out and "attn.out_proj" in name:
                torch.nn.init.zeros_(parameter)
            elif "ffn.w3" in name:
                torch.nn.init.zeros_(parameter)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _chunked_lm_loss(self, hidden_states, labels, *, loss_chunk_size: int, return_logits: bool):
        vocab_size = self.lm_head.weight.size(0)
        total_loss = hidden_states.new_zeros(())
        total_tokens = 0
        logits_chunks = [] if return_logits else None
        for start in range(0, hidden_states.size(1), loss_chunk_size):
            end = min(start + loss_chunk_size, hidden_states.size(1))
            logits_chunk = self.lm_head(hidden_states[:, start:end, :])
            labels_chunk = labels[:, start:end]
            total_loss = total_loss + F.cross_entropy(
                logits_chunk.reshape(-1, vocab_size).float(),
                labels_chunk.reshape(-1),
                reduction="sum",
            )
            total_tokens += labels_chunk.numel()
            if return_logits:
                logits_chunks.append(logits_chunk)
        logits = torch.cat(logits_chunks, dim=1) if return_logits else None
        return logits, total_loss / total_tokens

    def forward(self, input_ids, labels=None, *, return_logits=True, loss_chunk_size=0):
        hidden_states = self.backbone(input_ids)
        if labels is None:
            return self.lm_head(hidden_states), None

        loss_chunk_size = int(loss_chunk_size or 0)
        if loss_chunk_size > 0:
            if return_logits:
                return self._chunked_lm_loss(
                    hidden_states,
                    labels,
                    loss_chunk_size=loss_chunk_size,
                    return_logits=True,
                )
            loss = chunked_linear_cross_entropy(
                hidden_states,
                self.lm_head.weight,
                labels,
                chunk_size=loss_chunk_size,
            )
            return None, loss

        logits = self.lm_head(hidden_states)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1))
        return (logits if return_logits else None), loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 128
    model = HybridMOGTTransformerForCausalLM(
        vocab_size=1024,
        d_model=d_model,
        num_layers=4,
        num_heads=choose_num_heads(d_model, 4),
        r=16,
        pattern="alternating",
    ).to(device)
    model.backbone.configure_mogt(scan_impl="triton_hybrid", connection_impl="identity", connection_damping=1.0)
    x = torch.randint(0, 1024, (2, 128), device=device)
    y = torch.randint(0, 1024, (2, 128), device=device)
    _, loss = model(x, labels=y, return_logits=False, loss_chunk_size=64)
    print(f"loss={loss.item():.4f}")
