from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SIDE = 768           # resized side for matching
MIN_MATCHES = 16         # minimum filtered correspondences before estimating transform
RANSAC_REPROJ_THRESH = 4.0
ECC_ITERATIONS = 150
ECC_EPS = 1e-5
