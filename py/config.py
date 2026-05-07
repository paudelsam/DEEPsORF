import os
import torch
from pathlib import Path

# Tag the run
TAG = "train" 

# Training hyperparameters
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 512
EPOCHS = 30
K_FOLDS = 5
SEED = 9

ACCUMULATION_STEPS = 4
EFFECTIVE_BS = BATCH_SIZE * ACCUMULATION_STEPS

# Optimizer settings
MAX_LR = 2e-5
DIV_FACTOR = 25.0
FINAL_DIV_FACTOR = 1e+4
PCT_START = 0.2
WEIGHT_DECAY = 5e-3
MAX_NORM = 1.0

# Model hyperparameters
CONV_OUT = 128
GRU_OUT = 128
DROPOUT = 0.3
EMB_DROPOUT = 0.1

# Get the base dir
BASE_DIR = Path(__file__).resolve().parent.parent

# Dir paths
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "results" / TAG
CHECKPOINTS_DIR = RESULT_DIR / "checkpoints"
FINAL_MODEL_DIR = BASE_DIR / "final_model"
TB_DIR = RESULT_DIR / "tensor_board"

# File paths
SEQ_FILE_PATH = DATA_DIR / "train.csv"
TRAIN_EMB_FILE = DATA_DIR / "emb_train_dnabert.h5"
TEST_EMB_FILE = DATA_DIR / "emb_test_dnabert.h5"
TRAIN_ID_LABEL_FILE = DATA_DIR / "train_id_label.csv"
TEST_ID_LABEL_FILE = DATA_DIR / "test_id_label.csv"
LOSS_FILE= RESULT_DIR / "loss.csv"

CACHE_DIR = BASE_DIR / "cache"

# Early stopping
PATIENCE = 7
DELTA = 0.001

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Label smoothing
EPSILON = 0.05
CLASSES = 2