from pathlib import Path
import yaml
import cv2

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

raw_root = Path(cfg["paths"]["raw_root"])
processed_root = Path(cfg["paths"]["processed_root"])
img_size = int(cfg["project"]["image_size"])

terrains = ["agri", "barrenland", "grassland", "urban"]

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

count = 0

for terrain in terrains:
    s1_in = raw_root / terrain / "s1"
    s2_in = raw_root / terrain / "s2"

    s1_out = processed_root / terrain / "s1"
    s2_out = processed_root / terrain / "s2"

    s1_out.mkdir(parents=True, exist_ok=True)
    s2_out.mkdir(parents=True, exist_ok=True)

    s1_files = {p.name for p in s1_in.iterdir() if p.is_file()}
    s2_files = {p.name for p in s2_in.iterdir() if p.is_file()}
    common = sorted(s1_files & s2_files)

    for name in common:
        sar_img = cv2.imread(str(s1_in / name))
        opt_img = cv2.imread(str(s2_in / name))

        if sar_img is None or opt_img is None:
            print(f"Skipping unreadable file: {terrain}/{name}")
            continue

        sar_img = preprocess_sar(sar_img)
        opt_img = preprocess_optical(opt_img)

        cv2.imwrite(str(s1_out / name), sar_img)
        cv2.imwrite(str(s2_out / name), opt_img)
        count += 1

print(f"Preprocessed {count} matched pairs.")