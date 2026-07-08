import math
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import cfg
from dataset import TranslationDataset, tokenize_en, tokenize_zh_char, collate_fn
from model import Transformer

# 损失函数（将在 main 中初始化）
criterion = None

def noam_lr(step, d_model, warmup_steps):
    """Noam 学习率调度：先 warmup 线性增长，再按步长平方根衰减"""
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

def train_epoch(model, loader, optimizer, clip_grad, current_step):
    global criterion
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        src = batch['src']
        tgt_input = batch['tgt_input']
        tgt_output = batch['tgt_output']
        src_padding_mask = batch['src_mask']
        tgt_padding_mask = batch['tgt_mask']

        tgt_len = tgt_input.size(1)
        causal_mask = model.generate_square_subsequent_mask(tgt_len, src.device)

        optimizer.zero_grad()
        logits = model(src, tgt_input,
                       src_key_padding_mask=src_padding_mask,
                       tgt_mask=causal_mask,
                       tgt_key_padding_mask=tgt_padding_mask)
        loss = criterion(logits.permute(0, 2, 1), tgt_output)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        current_step += 1
        lr = noam_lr(current_step, cfg.d_model, cfg.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        total_loss += loss.item()
    return total_loss / len(loader), current_step

def evaluate(model, loader):
    global criterion
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            src = batch['src']
            tgt_input = batch['tgt_input']
            tgt_output = batch['tgt_output']
            src_padding_mask = batch['src_mask']
            tgt_padding_mask = batch['tgt_mask']

            tgt_len = tgt_input.size(1)
            causal_mask = model.generate_square_subsequent_mask(tgt_len, src.device)

            logits = model(src, tgt_input,
                           src_key_padding_mask=src_padding_mask,
                           tgt_mask=causal_mask,
                           tgt_key_padding_mask=tgt_padding_mask)
            loss = criterion(logits.permute(0, 2, 1), tgt_output)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    global criterion
    # 固定随机种子，保证可复现
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 加载完整数据集并保存词表
    full_dataset = TranslationDataset(cfg.data_path, tokenize_en, tokenize_zh_char,
                                      cfg.max_len, cfg.min_freq)
    with open("vocab.pkl", "wb") as f:
        pickle.dump({
            'src_vocab': full_dataset.src_vocab,
            'tgt_vocab': full_dataset.tgt_vocab,
            'src_pad_idx': full_dataset.src_pad_idx,
            'tgt_pad_idx': full_dataset.tgt_pad_idx,
            'src_sos_idx': full_dataset.src_sos_idx,
            'tgt_sos_idx': full_dataset.tgt_sos_idx,
            'src_eos_idx': full_dataset.src_eos_idx,
            'tgt_eos_idx': full_dataset.tgt_eos_idx,
            'src_unk_idx': full_dataset.src_unk_idx,
            'tgt_unk_idx': full_dataset.tgt_unk_idx,
        }, f)
    print(f"词表已保存：源词表 {len(full_dataset.src_vocab)}，目标词表 {len(full_dataset.tgt_vocab)}")

    # 划分训练/验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    print(f"训练集: {len(train_dataset)}，验证集: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # 初始化模型
    model = Transformer(
        src_vocab_size=len(full_dataset.src_vocab),
        tgt_vocab_size=len(full_dataset.tgt_vocab),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        max_len=cfg.max_pos_len,
        src_pad_idx=cfg.src_pad_idx,
        tgt_pad_idx=cfg.tgt_pad_idx
    ).to(cfg.device)

    # 损失函数（带标签平滑，ignore_index 忽略填充位）
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.tgt_pad_idx, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9,
                                 weight_decay=cfg.weight_decay)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    no_improve_epochs = 0
    current_step = 0

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        train_loss, current_step = train_epoch(model, train_loader, optimizer, cfg.clip_grad, current_step)
        val_loss = evaluate(model, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 早停与保存最佳模型（基于验证损失）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), "best_transformer.pt")
            print("  -> Best model saved (lowest validation loss).")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= cfg.early_stopping_patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    # 绘制损失曲线
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.show()

    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()