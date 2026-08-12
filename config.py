# config.py — All hyperparameters and settings in one place

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODEL_SAVE_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME    = "Fakeddit"
CSV_TRAIN       = os.path.join(DATA_DIR, "multimodal_train.tsv")
CSV_VAL         = os.path.join(DATA_DIR, "multimodal_validate.tsv")
CSV_TEST        = os.path.join(DATA_DIR, "multimodal_test_public.tsv")
IMAGE_DIR       = os.path.join(DATA_DIR, "images")

# Classification: 2 = Real/Fake
NUM_CLASSES     = 2
LABEL_COL       = "2_way_label"   # change to "3_way_label" or "6_way_label" if needed

# ─── Text Model (RoBERTa) ─────────────────────────────────────────────────────
ROBERTA_MODEL   = "roberta-base"
MAX_TEXT_LEN    = 128      # max tokens for headline + body
MAX_COMMENT_LEN = 64       # max tokens for comments

# ─── Image Model (ResNet-50) ──────────────────────────────────────────────────
IMAGE_SIZE      = 224      # ResNet expects 224×224

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 16
EPOCHS          = 5
LEARNING_RATE   = 2e-5
WEIGHT_DECAY    = 1e-2
DROPOUT         = 0.3
WARMUP_STEPS    = 100

# ─── Feature Dimensions ───────────────────────────────────────────────────────
TEXT_FEAT_DIM    = 768     # RoBERTa-base hidden size
IMAGE_FEAT_DIM   = 2048    # ResNet-50 penultimate layer
COMMENT_FEAT_DIM = 768     # RoBERTa-base hidden size
FUSED_DIM        = TEXT_FEAT_DIM + IMAGE_FEAT_DIM + COMMENT_FEAT_DIM  # 3584

# ─── Device ───────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
