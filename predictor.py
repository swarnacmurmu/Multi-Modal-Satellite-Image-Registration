# from pathlib import Path
# import cv2
# import yaml
# import torch
# import numpy as np

# from training.model import AffineRegistrationNet


# class Predictor:
#     def __init__(self):
#         with open("config.yaml", "r") as f:
#             cfg = yaml.safe_load(f)

#         self.image_size = int(cfg["project"]["image_size"])
#         self.outputs_dir = Path(cfg["paths"]["outputs_dir"])
#         self.outputs_dir.mkdir(parents=True, exist_ok=True)

#         ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / "best_registration_model.pt"

#         if not ckpt_path.exists():
#             raise FileNotFoundError(
#                 f"Model checkpoint not found at {ckpt_path}. Train the model first."
#             )

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.model = AffineRegistrationNet().to(self.device)
#         checkpoint = torch.load(ckpt_path, map_location=self.device)

#         if isinstance(checkpoint, dict)     and         "model_state_dict" in   checkpoint:
#             self.model.load_state_dict(checkpoint["model_state_dict"])
#         else:
#             self.model.load_state_dict(checkpoint)

#             self.model.eval()

#     def _prep(self, file_bytes):
#         arr = np.frombuffer(file_bytes, np.uint8)
#         img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

#         if img is None:
#             raise ValueError("Invalid uploaded image.")

#         img = cv2.resize(img, (self.image_size, self.image_size))
#         img = img.astype("float32") / 255.0

#         return img

#     def _to_uint8(self, img):
#         return np.clip(img * 255, 0, 255).astype(np.uint8)

#     def _make_overlay(self, fixed, moving):
#         fixed_u8 = self._to_uint8(fixed)
#         moving_u8 = self._to_uint8(moving)

#         fixed_color = cv2.cvtColor(fixed_u8, cv2.COLOR_GRAY2BGR)
#         moving_color = cv2.applyColorMap(moving_u8, cv2.COLORMAP_JET)

#         overlay = cv2.addWeighted(fixed_color, 0.6, moving_color, 0.4, 0)
#         return overlay

#     def predict(self, moving_bytes, fixed_bytes):
#         moving = self._prep(moving_bytes)
#         fixed = self._prep(fixed_bytes)

#         moving_t = torch.tensor(moving).unsqueeze(0).unsqueeze(0).to(self.device)
#         fixed_t = torch.tensor(fixed).unsqueeze(0).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             warped, theta_pred = self.model(moving_t, fixed_t)

#         warped_np = warped.squeeze().cpu().numpy()

#         input_sar_path = self.outputs_dir / "web_input_sar.png"
#         fixed_optical_path = self.outputs_dir / "web_fixed_optical.png"
#         warped_sar_path = self.outputs_dir / "web_warped_sar.png"
#         before_overlay_path = self.outputs_dir / "web_before_overlay.png"
#         after_overlay_path = self.outputs_dir / "web_after_overlay.png"

#         cv2.imwrite(str(input_sar_path), self._to_uint8(moving))
#         cv2.imwrite(str(fixed_optical_path), self._to_uint8(fixed))
#         cv2.imwrite(str(warped_sar_path), self._to_uint8(warped_np))

#         before_overlay = self._make_overlay(fixed, moving)
#         after_overlay = self._make_overlay(fixed, warped_np)

#         cv2.imwrite(str(before_overlay_path), before_overlay)
#         cv2.imwrite(str(after_overlay_path), after_overlay)

#         return {
#             "input_sar": f"/outputs/{input_sar_path.name}",
#             "fixed_optical": f"/outputs/{fixed_optical_path.name}",
#             "warped_sar": f"/outputs/{warped_sar_path.name}",
#             "before_overlay": f"/outputs/{before_overlay_path.name}",
#             "after_overlay": f"/outputs/{after_overlay_path.name}",
#             "theta": theta_pred.squeeze().cpu().numpy().tolist()
#         }















# from pathlib import Path
# import cv2
# import yaml
# import torch
# import numpy as np

# from training.model import AffineRegistrationNet


# class Predictor:
#     def __init__(self):
#         with open("config.yaml", "r") as f:
#             cfg = yaml.safe_load(f)

#         self.image_size = int(cfg["project"]["image_size"])
#         self.outputs_dir = Path(cfg["paths"]["outputs_dir"])
#         self.outputs_dir.mkdir(parents=True, exist_ok=True)

#         ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / "best_registration_model.pt"

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.model = AffineRegistrationNet().to(self.device)

#         checkpoint = torch.load(ckpt_path, map_location=self.device)
#         if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
#             self.model.load_state_dict(checkpoint["model_state_dict"])
#         else:
#             self.model.load_state_dict(checkpoint)

#         self.model.eval()

#     def _prep(self, file_bytes):
#         arr = np.frombuffer(file_bytes, np.uint8)
#         img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

#         if img is None:
#             raise ValueError("Invalid uploaded image.")

#         img = cv2.resize(img, (self.image_size, self.image_size))
#         img = img.astype("float32") / 255.0

#         return img

#     def _to_uint8(self, img):
#         return np.clip(img * 255, 0, 255).astype(np.uint8)

#     def _make_overlay(self, fixed, moving):
#         fixed_u8 = self._to_uint8(fixed)
#         moving_u8 = self._to_uint8(moving)

#         fixed_color = cv2.cvtColor(fixed_u8, cv2.COLOR_GRAY2BGR)
#         moving_color = cv2.applyColorMap(moving_u8, cv2.COLORMAP_JET)

#         return cv2.addWeighted(fixed_color, 0.6, moving_color, 0.4, 0)

#     def _synthetic_misalign(self, img):
#         h, w = img.shape

#         angle = 8
#         tx = 18
#         ty = -14

#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         M[0, 2] += tx
#         M[1, 2] += ty

#         moved = cv2.warpAffine(
#             img,
#             M,
#             (w, h),
#             flags=cv2.INTER_LINEAR,
#             borderMode=cv2.BORDER_REFLECT101
#         )

#         return moved

#     def predict(self, moving_bytes, fixed_bytes):
#         original_sar = self._prep(moving_bytes)
#         fixed = self._prep(fixed_bytes)

#         # Artificially misalign SAR to make before/after visible
#         misaligned_sar = self._synthetic_misalign(original_sar)

#         moving_t = torch.tensor(misaligned_sar).unsqueeze(0).unsqueeze(0).to(self.device)
#         fixed_t = torch.tensor(fixed).unsqueeze(0).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             warped, theta_pred = self.model(moving_t, fixed_t)

#         warped_np = warped.squeeze().cpu().numpy()

#         input_sar_path = self.outputs_dir / "web_input_sar.png"
#         misaligned_sar_path = self.outputs_dir / "web_misaligned_sar.png"
#         fixed_optical_path = self.outputs_dir / "web_fixed_optical.png"
#         warped_sar_path = self.outputs_dir / "web_warped_sar.png"
#         before_overlay_path = self.outputs_dir / "web_before_overlay.png"
#         after_overlay_path = self.outputs_dir / "web_after_overlay.png"

#         cv2.imwrite(str(input_sar_path), self._to_uint8(original_sar))
#         cv2.imwrite(str(misaligned_sar_path), self._to_uint8(misaligned_sar))
#         cv2.imwrite(str(fixed_optical_path), self._to_uint8(fixed))
#         cv2.imwrite(str(warped_sar_path), self._to_uint8(warped_np))

#         before_overlay = self._make_overlay(fixed, misaligned_sar)
#         after_overlay = self._make_overlay(fixed, warped_np)

#         cv2.imwrite(str(before_overlay_path), before_overlay)
#         cv2.imwrite(str(after_overlay_path), after_overlay)

#         return {
#             "input_sar": f"/outputs/{input_sar_path.name}",
#             "misaligned_sar": f"/outputs/{misaligned_sar_path.name}",
#             "fixed_optical": f"/outputs/{fixed_optical_path.name}",
#             "warped_sar": f"/outputs/{warped_sar_path.name}",
#             "before_overlay": f"/outputs/{before_overlay_path.name}",
#             "after_overlay": f"/outputs/{after_overlay_path.name}",
#             "theta": theta_pred.squeeze().cpu().numpy().tolist(),
#             "note": "The SAR image was synthetically shifted/rotated before registration to visualize correction."
#         }











from pathlib import Path
import cv2
import yaml
import torch
import numpy as np

from training.model import AffineRegistrationNet


class Predictor:
    def __init__(self):
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self.image_size = int(cfg["project"]["image_size"])
        self.outputs_dir = Path(cfg["paths"]["outputs_dir"])
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / "best_registration_model.pt"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AffineRegistrationNet().to(self.device)

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

    def _prep(self, file_bytes):
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Invalid uploaded image.")

        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype("float32") / 255.0
        return img

    def _to_uint8(self, img):
        return np.clip(img * 255, 0, 255).astype(np.uint8)

    def _synthetic_misalign(self, img):
        h, w = img.shape

        angle = 14
        tx = 28
        ty = -24

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty

        moved = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101
        )

        return moved

    def _red_green_overlay(self, fixed, moving):
        fixed_u8 = self._to_uint8(fixed)
        moving_u8 = self._to_uint8(moving)

        overlay = np.zeros((fixed_u8.shape[0], fixed_u8.shape[1], 3), dtype=np.uint8)

        # BGR format:
        # SAR = Red
        # Optical = Green
        overlay[:, :, 1] = fixed_u8       # green
        overlay[:, :, 2] = moving_u8      # red

        return overlay

    def _difference_heatmap(self, before_overlay, after_overlay):
        before_gray = cv2.cvtColor(before_overlay, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_overlay, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(before_gray, after_gray)

        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

        return heatmap

    def _marked_difference(self, fixed, before_moving, after_moving):
        fixed_u8 = self._to_uint8(fixed)
        before_u8 = self._to_uint8(before_moving)
        after_u8 = self._to_uint8(after_moving)

        before_diff = cv2.absdiff(fixed_u8, before_u8)
        after_diff = cv2.absdiff(fixed_u8, after_u8)

        improvement = cv2.absdiff(before_diff, after_diff)

        _, mask = cv2.threshold(improvement, 25, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        base = cv2.cvtColor(fixed_u8, cv2.COLOR_GRAY2BGR)

        # Mark changed regions in red
        base[mask > 0] = [0, 0, 255]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(base, contours, -1, (0, 255, 255), 1)

        return base

    def predict(self, moving_bytes, fixed_bytes):
        original_sar = self._prep(moving_bytes)
        fixed = self._prep(fixed_bytes)

        # Artificially misalign SAR for visible demo
        misaligned_sar = self._synthetic_misalign(original_sar)

        moving_t = torch.tensor(misaligned_sar).unsqueeze(0).unsqueeze(0).to(self.device)
        fixed_t = torch.tensor(fixed).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            warped, theta_pred = self.model(moving_t, fixed_t)

        warped_np = warped.squeeze().cpu().numpy()

        input_sar_path = self.outputs_dir / "web_input_sar.png"
        misaligned_sar_path = self.outputs_dir / "web_misaligned_sar.png"
        fixed_optical_path = self.outputs_dir / "web_fixed_optical.png"
        warped_sar_path = self.outputs_dir / "web_warped_sar.png"
        before_overlay_path = self.outputs_dir / "web_before_overlay.png"
        after_overlay_path = self.outputs_dir / "web_after_overlay.png"
        diff_heatmap_path = self.outputs_dir / "web_difference_heatmap.png"
        marked_difference_path = self.outputs_dir / "web_marked_difference.png"

        before_overlay = self._red_green_overlay(fixed, misaligned_sar)
        after_overlay = self._red_green_overlay(fixed, warped_np)

        diff_heatmap = self._difference_heatmap(before_overlay, after_overlay)
        marked_difference = self._marked_difference(fixed, misaligned_sar, warped_np)

        cv2.imwrite(str(input_sar_path), self._to_uint8(original_sar))
        cv2.imwrite(str(misaligned_sar_path), self._to_uint8(misaligned_sar))
        cv2.imwrite(str(fixed_optical_path), self._to_uint8(fixed))
        cv2.imwrite(str(warped_sar_path), self._to_uint8(warped_np))
        cv2.imwrite(str(before_overlay_path), before_overlay)
        cv2.imwrite(str(after_overlay_path), after_overlay)
        cv2.imwrite(str(diff_heatmap_path), diff_heatmap)
        cv2.imwrite(str(marked_difference_path), marked_difference)

        return {
            "input_sar": f"/outputs/{input_sar_path.name}",
            "misaligned_sar": f"/outputs/{misaligned_sar_path.name}",
            "fixed_optical": f"/outputs/{fixed_optical_path.name}",
            "warped_sar": f"/outputs/{warped_sar_path.name}",
            "before_overlay": f"/outputs/{before_overlay_path.name}",
            "after_overlay": f"/outputs/{after_overlay_path.name}",
            "difference_heatmap": f"/outputs/{diff_heatmap_path.name}",
            "marked_difference": f"/outputs/{marked_difference_path.name}",
            "theta": theta_pred.squeeze().cpu().numpy().tolist()
        }