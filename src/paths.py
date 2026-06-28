from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = PROJECT_ROOT / "figures"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return its path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
