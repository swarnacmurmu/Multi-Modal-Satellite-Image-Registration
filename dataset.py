from pathlib import Path
import json
import math
import random
import yaml
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RegistrationDataset(Dataset):
    def __init__(self, split="train"):
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self.processed_root = Path(cfg["paths"]["processed_root"])
        self.splits_dir = Path(cfg["paths"]["splits_dir"])
        self.image_size = int(cfg["project"]["image_size"])
        self.max_translation = float(cfg["project"]["max_translation"])
        self.max_rotation_deg = float(cfg["project"]["max_rotation_deg"])
        self.max_scale_change = float(cfg["project"]["max_scale_change"])

        with open(self.splits_dir / f"{split}.json", "r") as f:
            self.items = json.load(f)

        self.split = split

    def __len__(self):
        return len(self.items)

    def _load_pair(self, terrain, filename):
        sar_path = self.processed_root / terrain / "s1" / filename
        opt_path = self.processed_root / terrain / "s2" / filename

        sar = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
        optical = cv2.imread(str(opt_path), cv2.IMREAD_GRAYSCALE)

        if sar is None or optical is None:
            raise ValueError(f"Could not read pair: {terrain}/{filename}")

        sar = sar.astype(np.float32) / 255.0
        optical = optical.astype(np.float32) / 255.0

        return sar, optical

    def _sample_transform(self):
        tx = random.uniform(-self.max_translation, self.max_translation)
        ty = random.uniform(-self.max_translation, self.max_translation)
        angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
        scale = random.uniform(
            1.0 - self.max_scale_change,
            1.0 + self.max_scale_change
        )
        return tx, ty, angle, scale

    def _build_affine_matrix(self, tx, ty, angle_deg, scale):
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle) * scale
        sin_a = math.sin(angle) * scale

        theta = np.array([
            [cos_a, -sin_a, tx],
            [sin_a,  cos_a, ty]
        ], dtype=np.float32)

        return theta

    def _warp_image(self, img, theta):
        h, w = img.shape

        M = np.array([
            [theta[0, 0], theta[0, 1], theta[0, 2] * w / 2.0],
            [theta[1, 0], theta[1, 1], theta[1, 2] * h / 2.0]
        ], dtype=np.float32)

        warped = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101
        )

        return warped

    def __getitem__(self, idx):
        item = self.items[idx]
        terrain = item["terrain"]
        filename = item["filename"]

        sar, optical = self._load_pair(terrain, filename)

        tx, ty, angle, scale = self._sample_transform()
        theta_gt = self._build_affine_matrix(tx, ty, angle, scale)

        moving_sar = self._warp_image(sar, theta_gt)

        moving_sar = torch.tensor(moving_sar).unsqueeze(0)
        optical = torch.tensor(optical).unsqueeze(0)
        theta_gt = torch.tensor(theta_gt)

        return {
            "moving": moving_sar,
            "fixed": optical,
            "theta_gt": theta_gt,
            "terrain": terrain,
            "filename": filename
        }