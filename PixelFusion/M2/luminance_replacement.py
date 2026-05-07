from pathlib import Path
import os
import cv2
import numpy as np


LLVIP_ROOT = Path("/workspace/LLVIP")
FUSED_ROOT = LLVIP_ROOT / "fused_y_replace"

LIST_FILES = [
    LLVIP_ROOT / "llvip_rgb_train.txt",
    LLVIP_ROOT / "llvip_rgb_val.txt",
    LLVIP_ROOT / "llvip_rgb_test.txt",
]


def load_rgb_image(path):
    """Load a visible RGB image."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_ir_image(path):
    """Load a grayscale infrared image."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read IR image: {path}")

    return image


def fuse_luminance_replacement(rgb_image, ir_image):
    """
    Fuse RGB and IR images by replacing the Y channel
    of the RGB image with the infrared image.
    """
    if rgb_image.shape[:2] != ir_image.shape[:2]:
        raise ValueError(
            f"Shape mismatch: RGB {rgb_image.shape[:2]} vs IR {ir_image.shape[:2]}"
        )

    ycrcb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    ycrcb_image[:, :, 0] = ir_image

    fused_image = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2RGB)
    return fused_image


def save_rgb_image(path, image):
    """Save an RGB image as JPG."""
    path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(path), image_bgr)

    if not success:
        raise IOError(f"Could not save fused image: {path}")


def create_labels_symlink(split):
    """
    Create the labels link for the fused dataset.
    Labels are not copied because visible and infrared images
    have the same annotations.
    """
    source_labels = LLVIP_ROOT / "visible" / split / "labels"
    target_labels = FUSED_ROOT / split / "labels"

    if not source_labels.exists():
        raise FileNotFoundError(f"Labels folder not found: {source_labels}")

    if target_labels.exists() or target_labels.is_symlink():
        return

    target_labels.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source_labels, target_labels, target_is_directory=True)


def process_list(txt_path):
    with open(txt_path, "r") as file:
        lines = [line.strip() for line in file if line.strip()]

    for rgb_line in lines:
        rgb_path = Path(rgb_line)
        ir_path = Path(str(rgb_path).replace("/visible/", "/infrared/"))
        fused_path = Path(str(rgb_path).replace("/visible/", f"/{FUSED_ROOT.name}/"))

        rgb_image = load_rgb_image(rgb_path)
        ir_image = load_ir_image(ir_path)

        fused_image = fuse_luminance_replacement(rgb_image, ir_image)
        save_rgb_image(fused_path, fused_image)

    print(f"{txt_path.name}: {len(lines)} fused images created.")


if __name__ == "__main__":
    print(f"Saving to: {FUSED_ROOT}")

    create_labels_symlink("train")
    create_labels_symlink("test")

    for txt_path in LIST_FILES:
        process_list(txt_path)

    print("Luminance replacement fusion finished.")