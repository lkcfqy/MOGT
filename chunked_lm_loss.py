import torch
import torch.nn.functional as F


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, labels, chunk_size: int):
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must be [B, L, D]")
        if labels.shape != hidden_states.shape[:2]:
            raise ValueError("labels must be [B, L]")
        if weight.ndim != 2 or weight.shape[1] != hidden_states.shape[-1]:
            raise ValueError("weight must be [V, D]")

        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        bsz, seq_len, d_model = hidden_states.shape
        total_tokens = bsz * seq_len
        hidden_flat = hidden_states.reshape(total_tokens, d_model)
        labels_flat = labels.reshape(total_tokens)

        loss_sum = hidden_states.new_zeros((), dtype=torch.float32)
        weight_float = weight.float()
        with torch.no_grad():
            for start in range(0, total_tokens, chunk_size):
                end = min(start + chunk_size, total_tokens)
                logits = hidden_flat[start:end].float().matmul(weight_float.t())
                loss_sum = loss_sum + F.cross_entropy(
                    logits,
                    labels_flat[start:end],
                    reduction="sum",
                )

        ctx.save_for_backward(hidden_states, weight, labels)
        ctx.chunk_size = chunk_size
        ctx.total_tokens = total_tokens
        return loss_sum / total_tokens

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, weight, labels = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        total_tokens = ctx.total_tokens

        bsz, seq_len, d_model = hidden_states.shape
        hidden_flat = hidden_states.reshape(total_tokens, d_model)
        labels_flat = labels.reshape(total_tokens)

        grad_hidden_flat = torch.empty_like(hidden_flat)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        weight_float = weight.float()
        scale = grad_output.float() / float(total_tokens)

        for start in range(0, total_tokens, chunk_size):
            end = min(start + chunk_size, total_tokens)
            hidden_chunk = hidden_flat[start:end]
            labels_chunk = labels_flat[start:end]

            logits = hidden_chunk.float().matmul(weight_float.t())
            probs = torch.softmax(logits, dim=-1)
            probs[torch.arange(end - start, device=labels.device), labels_chunk] -= 1.0
            probs = probs * scale

            grad_hidden_flat[start:end] = probs.matmul(weight_float).to(hidden_states.dtype)
            grad_weight = grad_weight + probs.t().matmul(hidden_chunk.float())

        return grad_hidden_flat.view_as(hidden_states), grad_weight.to(weight.dtype), None, None


def chunked_linear_cross_entropy(hidden_states, weight, labels, *, chunk_size: int):
    """
    Memory-efficient CE for tied LM heads.

    The standard chunked loss still retains one autograd graph per chunk,
    including logits. This Function stores only hidden states, labels, and the
    LM weight, then recomputes logits chunk-by-chunk in backward.
    """
    return _ChunkedLinearCrossEntropy.apply(hidden_states, weight, labels, int(chunk_size))
