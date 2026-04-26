from pathlib import Path
import argparse
import cv2
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from training.model import AffineRegistrationNet
from training.losses import sobel_edges


def load_gray(path, size):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    img = cv2.resize(img, (size, size))
    img = img.astype("float32") / 255.0

    return img


def load_color(path, size):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def save_gray_as_color(img, path):
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path), img_color)


def save_rgb_image(img, path):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)


def load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def compute_ssim_score(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    )

    return float(ssim.mean().item())


def create_overlay(registered_gray, optical_rgb):
    registered_uint8 = (registered_gray * 255).clip(0, 255).astype(np.uint8)
    registered_rgb = cv2.cvtColor(registered_uint8, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(optical_rgb, 0.65, registered_rgb, 0.35, 0)

    return overlay


def run_prediction(moving_path, fixed_path):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    image_size = int(cfg["project"]["image_size"])
    outputs_dir = Path(cfg["paths"]["outputs_dir"]) / "inference"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / "best_registration_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For model input
    moving_gray = load_gray(moving_path, image_size)
    fixed_gray = load_gray(fixed_path, image_size)

    # For display
    moving_color = load_color(moving_path, image_size)
    fixed_color = load_color(fixed_path, image_size)

    moving_t = torch.tensor(moving_gray).unsqueeze(0).unsqueeze(0).to(device)
    fixed_t = torch.tensor(fixed_gray).unsqueeze(0).unsqueeze(0).to(device)

    model = AffineRegistrationNet().to(device)
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    with torch.no_grad():
        warped, theta_pred = model(moving_t, fixed_t)

    warped_np = warped.squeeze().cpu().numpy()

    overlay = create_overlay(warped_np, fixed_color)

    sar_path = outputs_dir / "sar_input.png"
    optical_path = outputs_dir / "optical_input.png"
    registered_path = outputs_dir / "registered_output.png"
    overlay_path = outputs_dir / "overlay_output.png"
    comparison_path = outputs_dir / "prediction_result.png"

    save_rgb_image(moving_color, sar_path)
    save_rgb_image(fixed_color, optical_path)
    save_gray_as_color(warped_np, registered_path)
    save_rgb_image(overlay, overlay_path)

    l1_loss = float(F.l1_loss(warped, fixed_t).item())

    edge_warped = sobel_edges(warped)
    edge_fixed = sobel_edges(fixed_t)
    edge_loss = float(F.l1_loss(edge_warped, edge_fixed).item())

    ssim_score = compute_ssim_score(warped, fixed_t)

    theta = theta_pred.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(moving_color)
    axes[0].set_title("SAR Input")

    axes[1].imshow(fixed_color)
    axes[1].set_title("Optical Reference")

    axes[2].imshow(warped_np, cmap="gray")
    axes[2].set_title("Registered SAR")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(comparison_path, dpi=150)
    plt.close()

    result = {
        "sar_image": "/outputs/inference/sar_input.png",
        "optical_image": "/outputs/inference/optical_input.png",
        "registered_image": "/outputs/inference/registered_output.png",
        "overlay_image": "/outputs/inference/overlay_output.png",
        "comparison_image": "/outputs/inference/prediction_result.png",
        "l1_loss": round(l1_loss, 6),
        "edge_loss": round(edge_loss, 6),
        "ssim_score": round(ssim_score, 6),
        "theta": theta.tolist()
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving", required=True)
    parser.add_argument("--fixed", required=True)
    args = parser.parse_args()

    result = run_prediction(args.moving, args.fixed)

    print("Saved prediction image to:", result["comparison_image"])
    print("Predicted theta:")
    print(result["theta"])


if __name__ == "__main__":
    main()