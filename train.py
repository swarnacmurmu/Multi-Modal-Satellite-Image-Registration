from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.dataset import RegistrationDataset
from training.model import AffineRegistrationNet
from training.losses import registration_loss

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    batch_size = int(cfg["project"]["batch_size"])
    epochs = int(cfg["project"]["epochs"])
    lr = float(cfg["project"]["learning_rate"])
    checkpoints_dir = Path(cfg["paths"]["checkpoints_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = RegistrationDataset("train")
    val_ds = RegistrationDataset("val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = AffineRegistrationNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_img = 0.0
        train_theta = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            moving = batch["moving"].to(device)
            fixed = batch["fixed"].to(device)
            theta_gt = batch["theta_gt"].to(device)

            optimizer.zero_grad()
            warped, theta_pred = model(moving, fixed)
            loss, img_loss, theta_loss = registration_loss(warped, fixed, theta_pred, theta_gt)
            loss.backward()
            optimizer.step()

            train_total += loss.item()
            train_img += img_loss.item()
            train_theta += theta_loss.item()

        train_total /= max(len(train_loader), 1)
        train_img /= max(len(train_loader), 1)
        train_theta /= max(len(train_loader), 1)

        model.eval()
        val_total = 0.0
        val_img = 0.0
        val_theta = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                moving = batch["moving"].to(device)
                fixed = batch["fixed"].to(device)
                theta_gt = batch["theta_gt"].to(device)

                warped, theta_pred = model(moving, fixed)
                loss, img_loss, theta_loss = registration_loss(warped, fixed, theta_pred, theta_gt)

                val_total += loss.item()
                val_img += img_loss.item()
                val_theta += theta_loss.item()

        val_total /= max(len(val_loader), 1)
        val_img /= max(len(val_loader), 1)
        val_theta /= max(len(val_loader), 1)

        print(f"\nEpoch {epoch}")
        print(f"Train Total Loss: {train_total:.6f}")
        print(f"Train Image Loss: {train_img:.6f}")
        print(f"Train Theta Loss: {train_theta:.6f}")
        print(f"Val Total Loss:   {val_total:.6f}")
        print(f"Val Image Loss:   {val_img:.6f}")
        print(f"Val Theta Loss:   {val_theta:.6f}")

        if val_total < best_val_loss:
            best_val_loss = val_total
            save_path = checkpoints_dir / "best_registration_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to: {save_path}")

if __name__ == "__main__":
    main()
