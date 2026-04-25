from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import uuid

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch

from backend.config import DEVICE, MAX_SIDE, MIN_MATCHES, RANSAC_REPROJ_THRESH, OUTPUT_DIR, ECC_ITERATIONS, ECC_EPS
from backend.utils.image_ops import (
    resize_keep_aspect,
    sar_to_matchable_gray,
    optical_to_matchable_gray,
    warp_image,
    make_overlay,
    draw_matches,
    save_rgb,
)


@dataclass
class AlignmentResult:
    homography: list[list[float]]
    num_raw_matches: int
    num_good_matches: int
    num_inliers: int
    match_confidence_mean: float
    method: str
    output_paths: dict


class SarOpticalAligner:
    def __init__(self, device: str = DEVICE):
        self.device = device
        self.matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()

    def _to_tensor(self, gray: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(gray).float() / 255.0
        return x.unsqueeze(0).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def _loftr_matches(self, sar_gray: np.ndarray, opt_gray: np.ndarray):
        sar_rs, sar_scale = resize_keep_aspect(sar_gray, MAX_SIDE)
        opt_rs, opt_scale = resize_keep_aspect(opt_gray, MAX_SIDE)

        batch = {
            "image0": self._to_tensor(sar_rs),
            "image1": self._to_tensor(opt_rs),
        }
        out = self.matcher(batch)
        mkpts0 = out["keypoints0"].detach().cpu().numpy()
        mkpts1 = out["keypoints1"].detach().cpu().numpy()
        confidence = out["confidence"].detach().cpu().numpy()

        if len(mkpts0) == 0:
            return mkpts0, mkpts1, confidence

        mkpts0[:, 0] /= sar_scale
        mkpts0[:, 1] /= sar_scale
        mkpts1[:, 0] /= opt_scale
        mkpts1[:, 1] /= opt_scale
        return mkpts0, mkpts1, confidence

    def _estimate_homography(self, mkpts0: np.ndarray, mkpts1: np.ndarray, confidence: np.ndarray):
        if len(mkpts0) < MIN_MATCHES:
            return None, None, 0

        conf_mask = confidence >= np.quantile(confidence, 0.45)
        mkpts0_f = mkpts0[conf_mask]
        mkpts1_f = mkpts1[conf_mask]
        conf_f = confidence[conf_mask]

        if len(mkpts0_f) < 8:
            return None, conf_mask, 0

        H, inlier_mask = cv2.findHomography(
            mkpts0_f.astype(np.float32),
            mkpts1_f.astype(np.float32),
            cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJ_THRESH,
        )
        if H is None or inlier_mask is None:
            return None, conf_mask, 0
        return (H, conf_mask, int(inlier_mask.ravel().sum()), mkpts0_f, mkpts1_f, conf_f, inlier_mask.ravel().astype(bool))

    def _ecc_refine(self, sar_gray: np.ndarray, opt_gray: np.ndarray, H_init: np.ndarray) -> np.ndarray:
        """
        Optional refinement using ECC over gradient-like grayscale images.
        ECC uses a warp from source -> target space.
        """
        h, w = opt_gray.shape[:2]
        sar_warped = cv2.warpPerspective(sar_gray, H_init, (w, h), flags=cv2.INTER_LINEAR)

        warp = np.eye(3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERATIONS, ECC_EPS)
        try:
            _, warp = cv2.findTransformECC(
                opt_gray.astype(np.float32) / 255.0,
                sar_warped.astype(np.float32) / 255.0,
                warp,
                cv2.MOTION_HOMOGRAPHY,
                criteria,
                None,
                1,
            )
            return warp @ H_init
        except cv2.error:
            return H_init

    def align(self, sar_rgb: np.ndarray, optical_rgb: np.ndarray) -> AlignmentResult:
        t0 = time.time()
        sar_gray = sar_to_matchable_gray(sar_rgb)
        opt_gray = optical_to_matchable_gray(optical_rgb)

        mkpts0, mkpts1, conf = self._loftr_matches(sar_gray, opt_gray)
        raw_matches = len(mkpts0)

        est = self._estimate_homography(mkpts0, mkpts1, conf)
        if est is None or est[0] is None:
            raise RuntimeError(
                "Not enough reliable correspondences were found. Try clearer image pairs, stronger overlap, or same-scene images."
            )

        H, conf_mask, num_inliers, mkpts0_f, mkpts1_f, conf_f, inlier_mask = est
        H = self._ecc_refine(sar_gray, opt_gray, H)

        warped_sar = warp_image(sar_rgb, H, optical_rgb.shape[:2])
        overlay = make_overlay(optical_rgb, warped_sar)
        matches_vis = draw_matches(optical_rgb, sar_rgb, mkpts0_f, mkpts1_f, inlier_mask)

        run_id = f"run_{uuid.uuid4().hex[:10]}"
        run_dir = Path(OUTPUT_DIR) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        save_rgb(str(run_dir / "warped_sar.png"), warped_sar)
        save_rgb(str(run_dir / "overlay.png"), overlay)
        save_rgb(str(run_dir / "matches.png"), matches_vis)
        save_rgb(str(run_dir / "optical.png"), optical_rgb)
        save_rgb(str(run_dir / "sar.png"), sar_rgb)

        return AlignmentResult(
            homography=H.tolist(),
            num_raw_matches=raw_matches,
            num_good_matches=len(mkpts0_f),
            num_inliers=num_inliers,
            match_confidence_mean=float(conf_f.mean()) if len(conf_f) else 0.0,
            method=f"LoFTR + RANSAC + ECC ({self.device})",
            output_paths={
                "run_dir": str(run_dir),
                "warped_sar": str(run_dir / "warped_sar.png"),
                "overlay": str(run_dir / "overlay.png"),
                "matches": str(run_dir / "matches.png"),
                "elapsed_sec": round(time.time() - t0, 3),
            },
        )
