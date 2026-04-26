from pathlib import Path
import yaml
import json
import random

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

processed_root = Path(cfg["paths"]["processed_root"])
splits_dir = Path(cfg["paths"]["splits_dir"])
splits_dir.mkdir(parents=True, exist_ok=True)

train_ratio = float(cfg["project"]["train_ratio"])
val_ratio = float(cfg["project"]["val_ratio"])
seed = int(cfg["project"]["seed"])

terrains = ["agri", "barrenland", "grassland", "urban"]

pairs = []

for terrain in terrains:
    s1_dir = processed_root / terrain / "s1"
    s2_dir = processed_root / terrain / "s2"

    s1_files = {p.name for p in s1_dir.iterdir() if p.is_file()}
    s2_files = {p.name for p in s2_dir.iterdir() if p.is_file()}
    common = sorted(s1_files & s2_files)

    for name in common:
        pairs.append({
            "terrain": terrain,
            "filename": name
        })

random.seed(seed)
random.shuffle(pairs)

n = len(pairs)
n_train = int(n * train_ratio)
n_val = int(n * val_ratio)

train_pairs = pairs[:n_train]
val_pairs = pairs[n_train:n_train + n_val]
test_pairs = pairs[n_train + n_val:]

with open(splits_dir / "train.json", "w") as f:
    json.dump(train_pairs, f, indent=2)

with open(splits_dir / "val.json", "w") as f:
    json.dump(val_pairs, f, indent=2)

with open(splits_dir / "test.json", "w") as f:
    json.dump(test_pairs, f, indent=2)

print("Total pairs:", len(pairs))
print("Train:", len(train_pairs))
print("Val:", len(val_pairs))
print("Test:", len(test_pairs))