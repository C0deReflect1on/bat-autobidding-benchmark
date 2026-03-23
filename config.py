from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FPA_DIR = DATA_DIR / "fpa"

FPA_CAMPAIGNS_TRAIN = DATA_FPA_DIR / "campaigns_fpa_filtered_train_final.csv"
FPA_CAMPAIGNS_TEST = DATA_FPA_DIR / "campaigns_fpa_filtered_test_final.csv"
FPA_STATS_TRAIN = DATA_FPA_DIR / "stats_fpa_filtered_train_final.csv"
FPA_STATS_TEST = DATA_FPA_DIR / "stats_fpa_filtered_test_final.csv"
