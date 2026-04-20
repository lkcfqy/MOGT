import json

# Throughput
throughput_data = {
    "lengths": lengths,
    "mogt_times": mogt_times,
    "attn_times": attn_times,
    "attn_oom_idx": attn_oom_idx
}

# Lifelong
lifelong_data = {
    "steps": steps,
    "adam_retention_A": adam_retention_A,
    "ewc_retention_A": ewc_retention_A,
    "mogt_retention_A": mogt_retention_A
}

# Scaling
scaling_data = {
    "parameter_counts": parameter_counts,
    "losses": losses
}

# Perplexity
perplexity_data = {
    "results": results
}
