from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
KEYPOINTS_DIR = OUTPUTS_DIR / "keypoints"
FEATURES_DIR = OUTPUTS_DIR / "features"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = OUTPUTS_DIR / "logs"


def ensure_outputs() -> None:
    """
    Ensure output directories exist. Call before writing artifacts.
    """
    for path in (
        OUTPUTS_DIR,
        KEYPOINTS_DIR,
        FEATURES_DIR,
        MODELS_DIR,
        PREDICTIONS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
