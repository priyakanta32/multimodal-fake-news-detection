# predict.py
# ─────────────────────────────────────────────────────────────────────────────
# Prediction module — imported by app.py for web interface
# Also runnable standalone: python predict.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import torch
from PIL import Image
from io import BytesIO
from transformers import RobertaTokenizer
from torchvision import transforms

from config import (
    MODEL_SAVE_DIR, ROBERTA_MODEL, MAX_TEXT_LEN,
    MAX_COMMENT_LEN, IMAGE_SIZE, DEVICE
)
from models.multimodal_model import MultimodalFakeNewsDetector

# ── Setup tokenizer and image transform ───────────────────────────────────────
tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)

img_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

LABELS = {0: "FAKE", 1: "REAL"}


def load_model():
    """Load model once and return — call this once at startup."""
    model = MultimodalFakeNewsDetector().to(DEVICE)
    ckpt  = os.path.join(MODEL_SAVE_DIR, "best_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    print(f"✅ Model loaded from {ckpt}")
    return model


def tokenize(text: str, max_len: int):
    enc = tokenizer(str(text), max_length=max_len, padding="max_length",
                    truncation=True, return_tensors="pt")
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


def predict(model, title: str, comment: str = "", image_path: str = None):
    """
    Predict whether news is Real or Fake.
    Args:
        model      : loaded MultimodalFakeNewsDetector (pass from app.py)
        title      : news headline text
        comment    : optional user comment
        image_path : optional local image path
    Returns:
        dict with label, confidence, fake_prob, real_prob (all floats/strings)
    """
    text_ids, text_mask = tokenize(title,   MAX_TEXT_LEN)
    com_ids,  com_mask  = tokenize(comment, MAX_COMMENT_LEN)

    # Load image if provided, else blank tensor
    if image_path and os.path.exists(image_path):
        try:
            img   = Image.open(image_path).convert("RGB")
            image = img_transform(img).unsqueeze(0).to(DEVICE)
        except Exception:
            image = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    else:
        image = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    with torch.no_grad():
        logits = model(text_ids, text_mask, com_ids, com_mask, image)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred   = int(probs.argmax())

    return {
        "label"      : LABELS[pred],
        "confidence" : round(float(probs[pred]) * 100, 1),
        "fake_prob"  : round(float(probs[0]) * 100, 1),
        "real_prob"  : round(float(probs[1]) * 100, 1),
    }


# ── Standalone mode ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   🔍 MULTIMODAL FAKE NEWS DETECTOR")
    print("   RoBERTa + ResNet-50 | Fakeddit Dataset")
    print("=" * 55)
    print("Loading model...\n")

    _model = load_model()
    print("\n✅ Ready! Type any headline to analyze.\n")

    while True:
        print("-" * 55)
        title = input("📰 Enter headline (or 'quit' to exit):\n> ").strip()

        if title.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break

        if not title:
            print("⚠️  Please enter a headline!")
            continue

        comment = input("\n💬 Comment (optional, Enter to skip):\n> ").strip()

        print("\n🔄 Analyzing...\n")
        result = predict(_model, title, comment)

        emoji = "🔴" if result["label"] == "FAKE" else "🟢"
        print("=" * 55)
        print(f"  RESULT     : {emoji} {result['label']}")
        print(f"  CONFIDENCE : {result['confidence']}%")
        print(f"  Fake prob  : {result['fake_prob']}%")
        print(f"  Real prob  : {result['real_prob']}%")
        print("=" * 55 + "\n")