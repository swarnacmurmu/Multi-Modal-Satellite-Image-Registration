from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from training.dataset import RegistrationDataset
from training.model import AffineRegistrationNet
from training.losses import registration_loss


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    checkpoints_dir = Path(cfg["paths"]["checkpoints_dir"])
    ckpt_path = checkpoints_dir / "best_registration_model.pt"
    batch_size = int(cfg["project"]["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_ds = RegistrationDataset("test")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = AffineRegistrationNet().to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    total = 0.0
    l1_total = 0.0
    edge_total = 0.0
    ssim_total = 0.0
    theta_total = 0.0

    with torch.no_grad():
        for batch in test_loader:
            moving = batch["moving"].to(device)
            fixed = batch["fixed"].to(device)
            theta_gt = batch["theta_gt"].to(device)

            warped, theta_pred = model(moving, fixed)

            loss, l1_loss, edge_loss, ssim_loss, theta_loss = registration_loss(
                warped,
                fixed,
                theta_pred,
                theta_gt
            )

            total += loss.item()
            l1_total += l1_loss.item()
            edge_total += edge_loss.item()
            ssim_total += ssim_loss.item()
            theta_total += theta_loss.item()

    n = max(len(test_loader), 1)

    print("Test Total Loss:", round(total / n, 6))
    print("Test L1 Loss:", round(l1_total / n, 6))
    print("Test Edge Loss:", round(edge_total / n, 6))
    print("Test SSIM Loss:", round(ssim_total / n, 6))
    print("Test Theta Loss:", round(theta_total / n, 6))


if __name__ == "__main__":
    main()