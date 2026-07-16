from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Specific file paths
TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
CLEANED_TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_train.csv"
PREPROCESSED_TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "preprocessed_train.csv"

# Models and Outputs paths
MODELS_DIR = BASE_DIR / "models"
VECTORIZERS_DIR = MODELS_DIR / "vectorizers"

OUTPUTS_DIR = BASE_DIR / "outputs"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
FEATURES_DIR = OUTPUTS_DIR / "features"

LOGS_DIR = BASE_DIR / "logs"
PROJECT_LOG = LOGS_DIR / "project.log"

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTORIZERS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
