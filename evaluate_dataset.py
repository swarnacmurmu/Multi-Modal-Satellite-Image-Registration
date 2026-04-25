"""
Dataset-level evaluation script.
Expected folder structure:

dataset/
  test/
    source/
      *.png
    target/
      *.png

It aligns every SAR source image to the corresponding target image and reports:
- SSIM between warped SAR-gray and optical-gray edge maps
- PSNR between warped and target edge maps
- NMI between warped and target edge maps

These are computed on modality-reduced edge representations, which is more meaningful than raw RGB-vs-SAR comparison.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import normalized_mutual_info_score

from backend.services.alignment_service import SarOpticalAligner
from backend.utils.image_ops import optical_to_matchable_gray, sar_to_matchable_gray


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def compute_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.uint8)
    b = b.astype(np.uint8)
    return {
        "ssim": float(ssim(a, b, data_range=255)),
        "psnr": float(psnr(a, b, data_range=255)),
        "nmi": float(normalized_mutual_info_score(a.flatten(), b.flatten())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_csv", type=str, default="outputs/eval_results.csv")
    args = parser.parse_args()

    src_dir = Path(args.dataset_root) / args.split / "source"
    tgt_dir = Path(args.dataset_root) / args.split / "target"
    src_files = sorted(src_dir.glob("*.png"))
    tgt_files = sorted(tgt_dir.glob("*.png"))
    assert len(src_files) == len(tgt_files), "source/target count mismatch"

    aligner = SarOpticalAligner()
    rows = []

    for src, tgt in zip(src_files, tgt_files):
        sar = read_rgb(src)
        opt = read_rgb(tgt)
        try:
            result = aligner.align(sar, opt)
            warped = np.array(Image.open(result.output_paths["warped_sar"]).convert("RGB"))
            warped_gray = sar_to_matchable_gray(warped)
            opt_gray = optical_to_matchable_gray(opt)
            m = compute_metrics(warped_gray, opt_gray)
            rows.append({
                "file": src.name,
                **m,
                "raw_matches": result.num_raw_matches,
                "good_matches": result.num_good_matches,
                "inliers": result.num_inliers,
                "elapsed_sec": result.output_paths["elapsed_sec"],
            })
            print(f"[OK] {src.name}: {m}")
        except Exception as e:
            rows.append({"file": src.name, "error": str(e)})
            print(f"[FAIL] {src.name}: {e}")

    df = pd.DataFrame(rows)
    Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.save_csv, index=False)
    print("\nSaved:", args.save_csv)
    if {"ssim", "psnr", "nmi"}.issubset(df.columns):
        print(df[["ssim", "psnr", "nmi"]].mean(numeric_only=True))


if __name__ == "__main__":
    main()
