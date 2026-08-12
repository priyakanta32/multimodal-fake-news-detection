# train.py
# ─────────────────────────────────────────────────────────────────────────────
# Training loop for Multimodal Fake News Detector
# Run: python train.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from config import (
    CSV_TRAIN, CSV_VAL, IMAGE_DIR, MODEL_SAVE_DIR, OUTPUT_DIR,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, WARMUP_STEPS, DEVICE
)
from utils.dataset import FakedditDataset
from models.multimodal_model import MultimodalFakeNewsDetector
from evaluate import evaluate_model


def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"🔧 Using device: {DEVICE}")

    # ── Datasets & Loaders ───────────────────────────────────────────────────
    # Set MAX_SAMPLES to None to use full dataset
    MAX_SAMPLES = 2000   # ← change to None for full dataset on Colab

    train_dataset = FakedditDataset(CSV_TRAIN, IMAGE_DIR, split="train", max_samples=MAX_SAMPLES)
    val_dataset   = FakedditDataset(CSV_VAL,   IMAGE_DIR, split="val",   max_samples=MAX_SAMPLES // 5 if MAX_SAMPLES else None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = MultimodalFakeNewsDetector().to(DEVICE)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()

    # ── Training History ──────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_acc = 0.0

    # ── Epoch Loop ───────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for batch in loop:
            text_ids  = batch["text_ids"].to(DEVICE)
            text_mask = batch["text_mask"].to(DEVICE)
            com_ids   = batch["com_ids"].to(DEVICE)
            com_mask  = batch["com_mask"].to(DEVICE)
            images    = batch["image"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(text_ids, text_mask, com_ids, com_mask, images)
            loss   = criterion(logits, labels)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # ── Validation ───────────────────────────────────────────────────
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, DEVICE)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Train Loss : {avg_train_loss:.4f}")
        print(f"   Val Loss   : {val_loss:.4f}")
        print(f"   Val Acc    : {val_acc:.4f}  |  Val F1: {val_f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(MODEL_SAVE_DIR, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"   ✅ Best model saved → {save_path}")

    # Save training history
    import json
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n🏁 Training complete! Best Val Accuracy: {best_val_acc:.4f}")
    return history


if __name__ == "__main__":
    train()