from pathlib import Path
import cv2
import numpy as np
import os


LLVIP_ROOT = Path("/workspace/LLVIP")

ALPHA = 0.4
FUSED_ROOT = LLVIP_ROOT / f"fused_alpha_{str(ALPHA).replace('.', '')}"

LIST_FILES = {
    "train": LLVIP_ROOT / "llvip_rgb_train.txt",
    "val": LLVIP_ROOT / "llvip_rgb_val.txt",
    "test": LLVIP_ROOT / "llvip_rgb_test.txt",
}


def load_rgb_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_ir_image_as_3ch(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read IR image: {path}")
    return np.stack([image, image, image], axis=-1)


def fuse_weighted_average(rgb_image, ir_image_3ch):
    if rgb_image.shape != ir_image_3ch.shape:
        raise ValueError(
            f"Shape mismatch: RGB {rgb_image.shape} vs IR {ir_image_3ch.shape}"
        )

    beta = 1.0 - ALPHA
    fused = ALPHA * rgb_image.astype(np.float32) + beta * ir_image_3ch.astype(np.float32)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused


def save_rgb_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(path), image_bgr)
    if not success:
        raise IOError(f"Could not save fused image: {path}")


def ensure_label_links():
    for split in ["train", "test"]:
        source_labels = LLVIP_ROOT / "visible" / split / "labels"
        target_labels = FUSED_ROOT / split / "labels"

        if not source_labels.exists():
            raise FileNotFoundError(f"Labels folder not found: {source_labels}")

        if target_labels.exists() or target_labels.is_symlink():
            continue

        target_labels.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source_labels, target_labels, target_is_directory=True)


def process_list(list_path):
    with open(list_path, "r") as file:
        lines = [line.strip() for line in file if line.strip()]

    for rgb_line in lines:
        rgb_path = Path(rgb_line)
        ir_path = Path(str(rgb_path).replace("/visible/", "/infrared/"))
        fused_path = Path(str(rgb_path).replace("/visible/", f"/{FUSED_ROOT.name}/"))

        rgb_image = load_rgb_image(rgb_path)
        ir_image_3ch = load_ir_image_as_3ch(ir_path)
        fused_image = fuse_weighted_average(rgb_image, ir_image_3ch)

        save_rgb_image(fused_path, fused_image)

    print(f"{list_path.name}: {len(lines)} fused images created.")


if __name__ == "__main__":
    print(f"Alpha = {ALPHA}")
    print(f"Saving to: {FUSED_ROOT}")

    ensure_label_links()

    for split_name, list_path in LIST_FILES.items():
        process_list(list_path)

    print("Weighted averaging fusion finished.")