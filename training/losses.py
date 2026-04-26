import torch
import torch.nn.functional as F


def sobel_edges(x):
    gx = torch.tensor(
        [[[[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]]],
        dtype=x.dtype,
        device=x.device
    )

    gy = torch.tensor(
        [[[[-1, -2, -1],
           [ 0,  0,  0],
           [ 1,  2,  1]]]],
        dtype=x.dtype,
        device=x.device
    )

    ex = F.conv2d(x, gx, padding=1)
    ey = F.conv2d(x, gy, padding=1)

    return torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)


def ssim_loss(x, y):
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

    return 1 - ssim.mean()


def registration_loss(warped, fixed, theta_pred, theta_gt):
    l1 = F.l1_loss(warped, fixed)

    edge_warped = sobel_edges(warped)
    edge_fixed = sobel_edges(fixed)
    edge = F.l1_loss(edge_warped, edge_fixed)

    ssim = ssim_loss(warped, fixed)

    theta = F.mse_loss(theta_pred, theta_gt)

    total = (
        0.30 * l1 +
        0.30 * edge +
        0.25 * ssim +
        5.00 * theta
    )

    return total, l1, edge, ssim, theta