"""Training loop with OneCycleLR scheduler and gradient clipping."""
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple

from . import DEVICE
from .tokenizer import SPTokenizer
from .model import Translator


def run_training(cfg, data: List[Tuple[str, str]]) -> None:
    """Run full training loop with all original features.
    
    Features:
        - 95/5 train/val split
        - AdamW optimizer with weight_decay=0.01
        - OneCycleLR scheduler with pct_start=0.3, cosine annealing
        - Gradient clipping with max_norm=1.0
        - Label smoothing
        - Model checkpoint saving
    
    Args:
        cfg: Config object with training hyperparameters
        data: List of (english, bengali) sentence pairs
    """
    if not data:
        return
    
    # Hold out the final 200 pairs as a disjoint evaluation set, then
    # shuffle the remainder with a fixed seed so the split is identical
    # across all five implementations and reproducible across runs.
    holdout = data[-200:]
    data = data[:-200]
    print(f"Split: holdout={len(holdout)} pool={len(data)} seed=20260815")
    random.seed(20260815)
    random.shuffle(data)
    split = int(len(data) * 0.95)
    train_data, val_data = data[:split], data[split:]
    
    # Remove existing checkpoint
    if os.path.exists("translator.pth"):
        os.remove("translator.pth")
    
    # Train tokenizers
    en_tok = SPTokenizer(cfg.vocab_size)
    bn_tok = SPTokenizer(cfg.vocab_size)
    en_tok.train([e for e, _ in train_data])
    bn_tok.train([b for _, b in train_data])
    
    # Initialize model
    m = Translator(cfg, en_tok.vocab_size, bn_tok.vocab_size).to(DEVICE)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=0.01)
    
    # OneCycleLR scheduler
    total_steps = cfg.epochs * (len(train_data) // cfg.batch_size + 1)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    print(f"Training (Data: {len(data)} pairs | Steps: {total_steps})")
    
    for ep in range(cfg.epochs):
        random.shuffle(train_data)
        m.train()
        total_loss = 0
        batch_count = 0
        
        for i in range(0, len(train_data), cfg.batch_size):
            batch = train_data[i:i + cfg.batch_size]
            if len(batch) < 2:
                continue
            
            # Prepare batch tensors with SOS and EOS tokens
            src = nn.utils.rnn.pad_sequence(
                [torch.tensor([1] + en_tok.encode(e) + [2]) for e, b in batch],
                batch_first=True
            ).to(DEVICE)
            
            trg = nn.utils.rnn.pad_sequence(
                [torch.tensor([1] + bn_tok.encode(b) + [2]) for e, b in batch],
                batch_first=True
            ).to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            logits = m(src, trg[:, :-1])
            
            # Cross-entropy loss with label smoothing
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                trg[:, 1:].reshape(-1),
                ignore_index=0,
                label_smoothing=cfg.label_smoothing
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            
            # Optimizer and scheduler step
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        # Epoch logging
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {ep + 1}/{cfg.epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.7f}")
    
    # Save model and tokenizers
    torch.save({
        "m": m.state_dict(),
        "en": en_tok,
        "bn": bn_tok,
        "cfg": cfg
    }, "translator.pth")
    print("Saved to 'translator.pth'")
