import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.get_device_name(0))
    print("CUDA version used by torch:", torch.version.cuda)
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    z = x @ y
    print("GPU test tensor shape:", tuple(z.shape))
else:
    print("PyTorch is not seeing your NVIDIA GPU yet.")
