# evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: accuracy, F1, confusion matrix, classification report
# Run standalone: python evaluate.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from tqdm import tqdm

from config import (
    CSV_TEST, IMAGE_DIR, MODEL_SAVE_DIR, OUTPUT_DIR,
    BATCH_SIZE, DEVICE, NUM_CLASSES
)
from utils.dataset import FakedditDataset
from models.multimodal_model import MultimodalFakeNewsDetector


def evaluate_model(model, loader, criterion, device):
    """Used during training to get val loss/acc/f1."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            text_ids  = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)
            com_ids   = batch["com_ids"].to(device)
            com_mask  = batch["com_mask"].to(device)
            images    = batch["image"].to(device)
            labels    = batch["label"].to(device)

            logits = model(text_ids, text_mask, com_ids, com_mask, images)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1


def full_evaluation():
    """Runs full test set evaluation with confusion matrix + report."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"🔧 Using device: {DEVICE}")

    test_dataset = FakedditDataset(CSV_TEST, IMAGE_DIR, split="test", max_samples=2000)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # Load best model
    model = MultimodalFakeNewsDetector().to(DEVICE)
    ckpt  = os.path.join(MODEL_SAVE_DIR, "best_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    print(f"✅ Loaded model from {ckpt}")

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            text_ids  = batch["text_ids"].to(DEVICE)
            text_mask = batch["text_mask"].to(DEVICE)
            com_ids   = batch["com_ids"].to(DEVICE)
            com_mask  = batch["com_mask"].to(DEVICE)
            images    = batch["image"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            logits = model(text_ids, text_mask, com_ids, com_mask, images)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc    = accuracy_score(all_labels, all_preds)
    f1     = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(
        all_labels, all_preds,
        target_names=["Fake", "Real"] if NUM_CLASSES == 2 else None
    )

    print(f"\n📊 TEST RESULTS")
    print(f"   Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"   F1 Score : {f1:.4f}")
    print(f"\n{report}")

    # Save report
    with open(os.path.join(OUTPUT_DIR, "test_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nF1: {f1:.4f}\n\n{report}")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm     = confusion_matrix(all_labels, all_preds)
    labels = ["Fake", "Real"] if NUM_CLASSES == 2 else [str(i) for i in range(NUM_CLASSES)]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix — Multimodal Fake News Detector")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.show()
    print(f"📈 Confusion matrix saved → {cm_path}")

    # ── Training Curves (if history exists) ───────────────────────────────────
    hist_path = os.path.join(OUTPUT_DIR, "training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        plot_training_curves(history)


def plot_training_curves(history: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, history["val_acc"], "g-o", label="Val Accuracy")
    ax2.plot(epochs, history["val_f1"],  "m-o", label="Val F1")
    ax2.set_title("Validation Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()

    plt.tight_layout()
    curve_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.show()
    print(f"📈 Training curves saved → {curve_path}")


if __name__ == "__main__":
    full_evaluation()
