import os
import re
import pandas as pd
import torch
from transformers import BertTokenizerFast
def load_and_prepare_data(data_dir="data", max_len=20, min_len=7, use_sample=True):
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, "raw_dataset.csv")
    
    if use_sample:
        raw_path = os.path.join(data_dir, "raw_dataset_sample.csv")
    
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    df = pd.DataFrame(lines, columns=["text"])
    df["text"] = df["text"].str.strip()
    df["text"] = df["text"].apply(lambda t: re.sub(r'[^a-z0-9\s]', '', t.lower()).strip())
    df["num_tokens"] = df["text"].str.split().apply(len)
    df = df[df["num_tokens"] >= min_len]
    df["tokens"] = df["text"].str.split().apply(lambda x: x[:max_len])
    df["text"] = df["tokens"].apply(lambda x: " ".join(x))

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokens = tokenizer(
        df["text"].tolist(),
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    torch.save(tokens, os.path.join(data_dir, "dataset_tokenized.pt"))
    return df, tokenizer, tokens
