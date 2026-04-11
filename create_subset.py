import os
import random
import shutil

RAW_ROOT = os.path.join("..", "data", "raw", "sentinel12")
OUTPUT_ROOT = os.path.join("..", "data", "processed", "subset")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MAX_PAIRS = 5000

random.seed(42)

def find_image_pairs(root):
    pairs = []
    for current_root, dirs, files in os.walk(root):
        if current_root.endswith("s1"):
            s1_dir = current_root
            s2_dir = current_root[:-2] + "s2"
            if os.path.exists(s2_dir):
                s1_files = sorted([
                    f for f in os.listdir(s1_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
                s2_files = set([
                    f for f in os.listdir(s2_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
                for file_name in s1_files:
                    if file_name in s2_files:
                        pairs.append((
                            os.path.join(s1_dir, file_name),
                            os.path.join(s2_dir, file_name),
                            file_name
                        ))
    return pairs

def ensure_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, "s1"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, split, "s2"), exist_ok=True)

def copy_pairs(pairs, split):
    for idx, (s1_path, s2_path, file_name) in enumerate(pairs):
        new_name = f"{idx:06d}_{file_name}"
        shutil.copy2(s1_path, os.path.join(OUTPUT_ROOT, split, "s1", new_name))
        shutil.copy2(s2_path, os.path.join(OUTPUT_ROOT, split, "s2", new_name))

def main():
    ensure_dirs()
    pairs = find_image_pairs(RAW_ROOT)

    if len(pairs) == 0:
        print("No pairs found.")
        return

    random.shuffle(pairs)
    pairs = pairs[:min(MAX_PAIRS, len(pairs))]

    total = len(pairs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    copy_pairs(test_pairs, "test")

    print("Subset created successfully")
    print("Train:", len(train_pairs))
    print("Val:", len(val_pairs))
    print("Test:", len(test_pairs))

if __name__ == "__main__":
    main()