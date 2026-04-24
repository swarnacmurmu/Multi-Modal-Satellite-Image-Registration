from pathlib import Path
import argparse
import cv2
import yaml
import torch
import matplotlib.pyplot as plt

from training.model import AffineRegistrationNet

def load_gray(path, size):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img, (size, size))
    img = img.astype("float32") / 255.0
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving", required=True, help="Path to moving SAR image")
    parser.add_argument("--fixed", required=True, help="Path to fixed optical image")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    image_size = int(cfg["project"]["image_size"])
    outputs_dir = Path(cfg["paths"]["outputs_dir"]) / "inference"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / "best_registration_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    moving = load_gray(args.moving, image_size)
    fixed = load_gray(args.fixed, image_size)

    moving_t = torch.tensor(moving).unsqueeze(0).unsqueeze(0).to(device)
    fixed_t = torch.tensor(fixed).unsqueeze(0).unsqueeze(0).to(device)

    model = AffineRegistrationNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        warped, theta_pred = model(moving_t, fixed_t)

    warped_np = warped.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(moving, cmap="gray")
    axes[0].set_title("Moving SAR")
    axes[1].imshow(fixed, cmap="gray")
    axes[1].set_title("Fixed Optical")
    axes[2].imshow(warped_np, cmap="gray")
    axes[2].set_title("Warped SAR")

    for ax in axes:
        ax.axis("off")

    save_path = outputs_dir / "prediction_result.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print("Saved prediction image to:", save_path)
    print("Predicted theta:")
    print(theta_pred.squeeze().cpu().numpy())

if __name__ == "__main__":
    main()