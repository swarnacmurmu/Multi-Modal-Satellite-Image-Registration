print("train.py started")

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

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = AffineRegistrationNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()

        train_total = 0.0
        train_l1 = 0.0
        train_edge = 0.0
        train_ssim = 0.0
        train_theta = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            moving = batch["moving"].to(device, non_blocking=True)
            fixed = batch["fixed"].to(device, non_blocking=True)
            theta_gt = batch["theta_gt"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                warped, theta_pred = model(moving, fixed)
                loss, l1_loss, edge_loss, ssim_l, theta_loss = registration_loss(
                    warped,
                    fixed,
                    theta_pred,
                    theta_gt
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_total += loss.item()
            train_l1 += l1_loss.item()
            train_edge += edge_loss.item()
            train_ssim += ssim_l.item()
            train_theta += theta_loss.item()

        train_total /= max(len(train_loader), 1)
        train_l1 /= max(len(train_loader), 1)
        train_edge /= max(len(train_loader), 1)
        train_ssim /= max(len(train_loader), 1)
        train_theta /= max(len(train_loader), 1)

        model.eval()

        val_total = 0.0
        val_l1 = 0.0
        val_edge = 0.0
        val_ssim = 0.0
        val_theta = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                moving = batch["moving"].to(device, non_blocking=True)
                fixed = batch["fixed"].to(device, non_blocking=True)
                theta_gt = batch["theta_gt"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    warped, theta_pred = model(moving, fixed)
                    loss, l1_loss, edge_loss, ssim_l, theta_loss = registration_loss(
                        warped,
                        fixed,
                        theta_pred,
                        theta_gt
                    )

                val_total += loss.item()
                val_l1 += l1_loss.item()
                val_edge += edge_loss.item()
                val_ssim += ssim_l.item()
                val_theta += theta_loss.item()

        val_total /= max(len(val_loader), 1)
        val_l1 /= max(len(val_loader), 1)
        val_edge /= max(len(val_loader), 1)
        val_ssim /= max(len(val_loader), 1)
        val_theta /= max(len(val_loader), 1)

        scheduler.step(val_total)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch}")
        print(f"Learning Rate:    {current_lr:.8f}")
        print(f"Train Total Loss: {train_total:.6f}")
        print(f"Train L1 Loss:    {train_l1:.6f}")
        print(f"Train Edge Loss:  {train_edge:.6f}")
        print(f"Train SSIM Loss:  {train_ssim:.6f}")
        print(f"Train Theta Loss: {train_theta:.6f}")
        print(f"Val Total Loss:   {val_total:.6f}")
        print(f"Val L1 Loss:      {val_l1:.6f}")
        print(f"Val Edge Loss:    {val_edge:.6f}")
        print(f"Val SSIM Loss:    {val_ssim:.6f}")
        print(f"Val Theta Loss:   {val_theta:.6f}")

        if val_total < best_val_loss:
            best_val_loss = val_total
            save_path = checkpoints_dir / "best_registration_model.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                save_path
            )

            print(f"Saved best model to: {save_path}")


if __name__ == "__main__":
    main()