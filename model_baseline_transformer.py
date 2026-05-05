import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from chunked_lm_loss import chunked_linear_cross_entropy


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        x_float = x.float()
        out = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return out.to(dtype=x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def choose_num_heads(d_model: int, requested: int = 0) -> int:
    if requested > 0:
        if d_model % requested != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={requested}")
        return requested

    for candidate in (16, 12, 8, 6, 4, 2, 1):
        if d_model % candidate == 0:
            return candidate
    return 1


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _rope_cache(seq_len: int, head_dim: int, device, dtype, theta: float):
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().repeat_interleave(2, dim=-1).view(1, 1, seq_len, head_dim)
    sin = freqs.sin().repeat_interleave(2, dim=-1).view(1, 1, seq_len, head_dim)
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply_rope(q: torch.Tensor, k: torch.Tensor, theta: float = 10000.0):
    seq_len = q.size(-2)
    head_dim = q.size(-1)
    cos, sin = _rope_cache(seq_len, head_dim, q.device, q.dtype, theta)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_theta: float = 10000.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_theta = rope_theta

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.rope_theta and self.rope_theta > 0:
            q, k = apply_rope(q, k, theta=self.rope_theta)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, rope_theta: float):
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, rope_theta=rope_theta)
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        hidden_dim = int(8 * d_model / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    rope_theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.gradient_checkpointing = False
        self.checkpoint_every_n = 1

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        use_checkpoint = self.training and self.gradient_checkpointing
        checkpoint_every_n = max(1, int(self.checkpoint_every_n))
        for block_idx, block in enumerate(self.blocks):
            if use_checkpoint and (block_idx % checkpoint_every_n == 0):
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm_f(x)


class TransformerForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.transformer = TransformerBackbone(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            rope_theta=rope_theta,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.transformer.embedding.weight
        self.apply(self._init_weights)

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

        loss = total_loss / total_tokens
        logits = torch.cat(logits_chunks, dim=1) if return_logits else None
        return logits, loss

    def forward(self, input_ids, labels=None, *, return_logits=True, loss_chunk_size=0):
        hidden_states = self.transformer(input_ids)
        logits = None
        loss = None
        loss_chunk_size = int(loss_chunk_size or 0)

        if labels is not None and loss_chunk_size > 0:
            if return_logits:
                logits, loss = self._chunked_lm_loss(
                    hidden_states,
                    labels,
                    loss_chunk_size=loss_chunk_size,
                    return_logits=True,
                )
            else:
                loss = chunked_linear_cross_entropy(
                    hidden_states,
                    self.lm_head.weight,
                    labels,
                    chunk_size=loss_chunk_size,
                )
        else:
            logits = self.lm_head(hidden_states)
            if labels is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1))
            if not return_logits:
                logits = None

        return logits, loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 50257
    d_model = 768
    num_heads = choose_num_heads(d_model, 12)
    model = TransformerForCausalLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=12,
        num_heads=num_heads,
    ).to(device)
    params = sum(param.numel() for param in model.parameters())
    print(f"params={params / 1e6:.2f}M heads={num_heads}")
    x = torch.randint(0, vocab_size, (2, 256), device=device)
    y = torch.randint(0, vocab_size, (2, 256), device=device)
    _, loss = model(x, labels=y, return_logits=False, loss_chunk_size=128)
    print(f"loss={loss.item():.4f}")
