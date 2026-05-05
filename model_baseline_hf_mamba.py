import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MambaConfig, MambaModel


class HFMambaForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 2,
        state_size: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
    ):
        super().__init__()
        config = MambaConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            state_size=state_size,
            num_hidden_layers=num_layers,
            expand=expand,
            conv_kernel=conv_kernel,
            initializer_range=0.02,
            residual_in_fp32=True,
            use_cache=False,
            tie_word_embeddings=True,
        )
        self.backbone = MambaModel(config)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.backbone.embeddings.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        hidden_states = self.backbone(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}
