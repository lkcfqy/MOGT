import os
import shutil
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

def collate_fn(batch):
    """
    DataLoader 的整理回调接口。负责将 length+1 的原序列，剥离为 x 和 target y。
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    # x 为预测用的输入 (舍弃最末一个 token)
    x = input_ids[:, :-1].contiguous()
    # y 为损失函数的 Ground Truth (错位对齐，舍弃最前一个 token)
    y = input_ids[:, 1:].contiguous()
    return x, y


def _processed_cache_path(context_length, tokenizer_name):
    cache_root = Path(os.environ.get("MOGT_DATA_CACHE_DIR", "./dataset_cache")).expanduser()
    safe_tokenizer_name = tokenizer_name.replace("/", "_")
    return cache_root / f"wikitext-103-raw-v1_{safe_tokenizer_name}_ctx{context_length}"


def _prepare_lm_datasets(tokenizer, seq_len, num_workers):
    def tokenize_function(examples):
        # 对原始自然语言文本进行 Token 分词，无需 pad，因为后续会将其全部缝合
        return tokenizer(examples["text"])

    def group_texts(examples):
        # 1. 将该 batch 中所有的 examples 的 tokens 扁平化，首尾拼接成一条大长龙
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # 2. 如果总长度小于单个 sequence 需求，则直接舍弃（概率极低），否则整数倍截断
        if total_length >= seq_len:
            total_length = (total_length // seq_len) * seq_len

        # 3. 再以 seq_len 大小等距将大长龙切断，打包回 result 对象返回给 dataset
        result = {
            k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated_examples.items()
        }
        return result

    print("🚀 [2/3] 正在拉取大规模基座数据: WikiText-103...")
    raw_datasets = load_dataset("wikitext", "wikitext-103-raw-v1")

    # Map 1: 开始并行多线程分词
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=["text"],
        desc="并行 Tokenizing 数据列",
    )

    # Map 2: 进行序列无缝拼接切分
    return tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
        desc=f"将 Token 序列切分为长度 {seq_len} 的定长序列",
    )

def get_dataloaders(context_length=2048, batch_size=8, num_workers=4):
    """
    流式拉取、清洗并处理 WikiText-103 数据集。
    结合 GPT2TokenizerFast 将所有文本首尾相连，随后利用固定 Context Length 的滑动窗口切分大块。
    """

    print("🚀 [1/3] 正在拉取与实例化 GPT-2 词表 (Tokenizer)...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 我们需要在最终迭代时得到长度为 context_length 的 x(由前到后) 和 y(向后错位一维)
    # 因此内部处理切块大小设定为 seq_len = context_length + 1
    seq_len = context_length + 1

    processed_cache_path = _processed_cache_path(
        context_length=context_length,
        tokenizer_name=tokenizer.name_or_path,
    )

    if processed_cache_path.exists():
        print(f"⚡ [2/3] 命中本地序列缓存: {processed_cache_path}")
        lm_datasets = load_from_disk(str(processed_cache_path))
    else:
        lm_datasets = _prepare_lm_datasets(tokenizer=tokenizer, seq_len=seq_len, num_workers=num_workers)
        processed_cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_cache_path = processed_cache_path.with_name(f"{processed_cache_path.name}.tmp-{os.getpid()}")
        if tmp_cache_path.exists():
            shutil.rmtree(tmp_cache_path)
        print(f"💾 [2/3] 正在将处理后的序列缓存到本地: {processed_cache_path}")
        lm_datasets.save_to_disk(str(tmp_cache_path))
        os.replace(tmp_cache_path, processed_cache_path)

    # 将输出设定为纯 PyTorch 张量格式
    lm_datasets.set_format(type="torch", columns=["input_ids"])

    print(f"🚀 [3/3] 数据清洗完成，正在组装 PyTorch DataLoader。Batch Size: {batch_size}, Ctx: {context_length}")

    train_loader = DataLoader(
        lm_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        lm_datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, tokenizer.vocab_size

if __name__ == "__main__":
    # 本地跑这个文件时的一键测试
    print("-----------------------------------------------------------")
    print("🛠️ MOGT WikiText-103 Dataloader 测试程序")
    train_dl, val_dl, vocab_size = get_dataloaders(context_length=256, batch_size=2)
    print(f"Vocab Size 判定: {vocab_size}")

    for idx, (x, y) in enumerate(train_dl):
        print(f"\\nBatch {idx+1} 测试：")
        print(f"Input X shape: {x.shape}")
        print(f"Target Y shape: {y.shape}")
        print(f"X 第一条样本前 5 个 token: {x[0, :5].tolist()}")
        print(f"Y 第一条样本前 5 个 token: {y[0, :5].tolist()} (应全部错位1个位置)")
        break # 测试一次即刻跳出
