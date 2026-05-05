import torch
import torch.nn as nn
from torch.optim import AdamW
from model_mogt import MOGTForCausalLM

def dummy_data_stream(vocab_size, batch_size=4, seq_len=128):
    while True:
        x = torch.randint(1, vocab_size, (batch_size, seq_len))
        y = torch.randint(1, vocab_size, (batch_size, seq_len))
        yield x, y

def run_sanity_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 50257
    print(f"Sanity Check: MOGT 10M Model on {device}")
    model = MOGTForCausalLM(vocab_size=vocab_size, d_model=256, num_layers=4, r=16).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    loss_fct = nn.CrossEntropyLoss()
    model.train()

    data_gen = dummy_data_stream(vocab_size)

    steps = 40
    for i in range(steps):
        x, y = next(data_gen)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = loss_fct(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (i+1) % 5 == 0 or i == 0:
            print(f"Step {i+1:02d}/{steps} - Loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_sanity_check()
