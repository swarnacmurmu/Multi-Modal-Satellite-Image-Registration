from pathlib import Path
import yaml
import cv2

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

raw_root = Path(cfg["paths"]["raw_root"])
processed_root = Path(cfg["paths"]["processed_root"])
img_size = int(cfg["project"]["image_size"])

terrains = ["agri", "barrenland", "grassland", "urban"]
valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def preprocess_sar(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (img_size, img_size))
    return img

def preprocess_optical(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (img_size, img_size))
    return img

total = 0

for terrain in terrains:
    s1_in = raw_root / terrain / "s1"
    s2_in = raw_root / terrain / "s2"

    s1_out = processed_root / terrain / "s1"
    s2_out = processed_root / terrain / "s2"

    s1_out.mkdir(parents=True, exist_ok=True)
    s2_out.mkdir(parents=True, exist_ok=True)

    s1_files = sorted([p for p in s1_in.iterdir() if p.suffix.lower() in valid_exts])
    s2_files = sorted([p for p in s2_in.iterdir() if p.suffix.lower() in valid_exts])

    pair_count = min(len(s1_files), len(s2_files))

    for i in range(pair_count):
        sar_path = s1_files[i]
        opt_path = s2_files[i]

        sar_img = cv2.imread(str(sar_path))
        opt_img = cv2.imread(str(opt_path))

        if sar_img is None or opt_img is None:
            print(f"Skipping unreadable pair: {sar_path.name}, {opt_path.name}")
            continue

        sar_img = preprocess_sar(sar_img)
        opt_img = preprocess_optical(opt_img)

        out_name = f"{terrain}_{i:05d}.png"

        cv2.imwrite(str(s1_out / out_name), sar_img)
        cv2.imwrite(str(s2_out / out_name), opt_img)

        total += 1

    print(f"{terrain}: preprocessed {pair_count} pairs")

print(f"\nTotal preprocessed pairs: {total}")