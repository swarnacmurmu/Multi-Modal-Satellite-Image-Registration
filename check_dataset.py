from pathlib import Path
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

raw_root = Path(cfg["paths"]["raw_root"])
terrains = ["agri", "barrenland", "grassland", "urban"]

valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

total_pairs = 0

for terrain in terrains:
    s1_dir = raw_root / terrain / "s1"
    s2_dir = raw_root / terrain / "s2"

    s1_files = sorted([p for p in s1_dir.iterdir() if p.suffix.lower() in valid_exts])
    s2_files = sorted([p for p in s2_dir.iterdir() if p.suffix.lower() in valid_exts])

    pair_count = min(len(s1_files), len(s2_files))
    total_pairs += pair_count

    print(f"{terrain}:")
    print(f"  s1 files = {len(s1_files)}")
    print(f"  s2 files = {len(s2_files)}")
    print(f"  usable pairs by order = {pair_count}")

    if pair_count > 0:
        print(f"  example pair:")
        print(f"    SAR:     {s1_files[0].name}")
        print(f"    Optical: {s2_files[0].name}")

print("\nTotal usable pairs across all terrains:", total_pairs)