from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


def read_image_bytes(file_bytes: bytes, color: bool = True) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imdecode(arr, flag)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid PNG/JPG image.")
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_keep_aspect(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale == 1.0:
        return img.copy(), 1.0
    resized = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    min_v, max_v = float(x.min()), float(x.max())
    if max_v - min_v < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - min_v) / (max_v - min_v)


def sar_to_matchable_gray(sar_rgb_or_gray: np.ndarray) -> np.ndarray:
    """
    Convert SAR image into a smoother, edge-rich grayscale representation.
    This reduces speckle and narrows the modality gap before matching.
    """
    if sar_rgb_or_gray.ndim == 3:
        gray = cv2.cvtColor(sar_rgb_or_gray, cv2.COLOR_RGB2GRAY)
    else:
        gray = sar_rgb_or_gray.copy()

    gray = gray.astype(np.float32)
    gray = np.log1p(gray)                      # log compression helps speckle-heavy SAR
    gray = normalize_01(gray)
    gray = (gray * 255).astype(np.uint8)
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = normalize_01(mag)
    return (mag * 255).astype(np.uint8)


def optical_to_matchable_gray(opt_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(opt_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = normalize_01(mag)
    return (mag * 255).astype(np.uint8)


def warp_image(img: np.ndarray, H: np.ndarray, out_shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = out_shape_hw
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def make_overlay(base_rgb: np.ndarray, warped_sar_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = base_rgb.astype(np.float32)
    sar = warped_sar_rgb.astype(np.float32)
    overlay = cv2.addWeighted(base, 1.0 - alpha, sar, alpha, 0)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_matches(opt_img: np.ndarray, sar_img: np.ndarray, mkpts0: np.ndarray, mkpts1: np.ndarray, inlier_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Draw filtered matches. mkpts0 belongs to SAR, mkpts1 belongs to optical.
    """
    sar_vis = sar_img if sar_img.ndim == 3 else cv2.cvtColor(sar_img, cv2.COLOR_GRAY2RGB)
    opt_vis = opt_img if opt_img.ndim == 3 else cv2.cvtColor(opt_img, cv2.COLOR_GRAY2RGB)
    h1, w1 = sar_vis.shape[:2]
    h2, w2 = opt_vis.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = sar_vis
    canvas[:h2, w1:w1 + w2] = opt_vis

    if inlier_mask is None:
        inlier_mask = np.ones(len(mkpts0), dtype=bool)

    for i, (p0, p1) in enumerate(zip(mkpts0, mkpts1)):
        c = (0, 255, 0) if bool(inlier_mask[i]) else (255, 80, 80)
        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0] + w1)), int(round(p1[1]))
        cv2.circle(canvas, (x0, y0), 3, c, -1)
        cv2.circle(canvas, (x1, y1), 3, c, -1)
        cv2.line(canvas, (x0, y0), (x1, y1), c, 1)
    return canvas


def save_rgb(path: str, img: np.ndarray) -> None:
    Image.fromarray(img).save(path)
