import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch(model, loader, optimizer, criterion, device="cpu"):
    model.train()
    total_loss = 0
    for X, Y in tqdm(loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
