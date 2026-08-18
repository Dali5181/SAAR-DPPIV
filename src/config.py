"""Global configuration for the SAAR-DPPIV pipeline."""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

RANDOM_SEED = 42
TEST_RATIO = 0.2

# CD-HIT-style sequence de-duplication threshold (see cluster_split.py).
# 80% identity, matching scripts/expand_merge_dataset.py in the full
# development history and the manuscript text. See cluster_split.py for a
# disclosed limitation: a later data merge was not re-deduplicated.
CDHIT_THRESHOLD = 0.8

RBM_N_COMPONENTS = 128
RBM_N_ITER = 50
RBM_LEARNING_RATE = 0.01
RBM_N_GENERATE = 300

ESM2_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
ESM2_EMBED_DIM = 480

TREE_MODELS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]
TOP_K_RANKING_FEATURES = 300

# Combined-score fusion weights (see src/models/pipeline_scoring.py and
# README.md "Fusion weight selection" for the grid-search evidence behind
# this choice).
W_PACTIVE = 0.65
W_RANK = 0.35
