import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineRegistrationNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
        )

        self._init_final_layer()

    def _init_final_layer(self):
        nn.init.zeros_(self.regressor[-1].weight)

        identity = torch.tensor(
            [1, 0, 0, 0, 1, 0],
            dtype=torch.float32
        )

        with torch.no_grad():
            self.regressor[-1].bias.copy_(identity)

    def forward(self, moving, fixed):
        x = torch.cat([moving, fixed], dim=1)

        theta = self.regressor(self.features(x)).view(-1, 2, 3)

        grid = F.affine_grid(theta, moving.size(), align_corners=False)

        warped = F.grid_sample(
            moving,
            grid,
            align_corners=False,
            mode="bilinear",
            padding_mode="border"
        )

        return warped, theta