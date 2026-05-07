import json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]  # LateFusion/
IMG_DIR = Path("/workspace/LLVIP/visible/test/images")
LBL_DIR = Path("/workspace/LLVIP/visible/test/labels")
OUT_JSON = ROOT/"M1_Union"/"llvip_test_gt_coco.json"

# COCO category: you used category_id=1 in predictions.json
CATEGORIES = [{"id": 1, "name": "person"}]


def yolo_to_coco_bbox(line, img_w, img_h):
    """
    Convert the yolo labels to match the predictions in COCO style.
    YOLO: cls cx cy w h (normalized in [0,1])
    COCO: [x_min, y_min, box_w, box_h] in pixels
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls, cx, cy, bw, bh = parts
    cx, cy, bw, bh = map(float, (cx, cy, bw, bh))

    x = (cx - bw / 2.0) * img_w
    y = (cy - bh / 2.0) * img_h
    w = bw * img_w
    h = bh * img_h

    # clamp minimally
    x = max(0.0, x); y = max(0.0, y)
    w = max(0.0, w); h = max(0.0, h)
    return [x, y, w, h]

def main():
    assert IMG_DIR.exists(), f"Missing images dir: {IMG_DIR}"
    assert LBL_DIR.exists(), f"Missing labels dir: {LBL_DIR}"

    images = []
    annotations = []
    ann_id = 1

    # We iterate over images, and look for same-stem label
    img_paths = sorted(list(IMG_DIR.glob("*.jpg")))
    if not img_paths:
        raise SystemExit(f"No .jpg images found in {IMG_DIR}")

    for img_path in img_paths:
        stem = img_path.stem  # e.g. 190001
        img_id = int(stem) if stem.isdigit() else stem  # keep robust

        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h
        })

        lbl_path = LBL_DIR / f"{stem}.txt"
        if not lbl_path.exists():
            # no labels => background image (allowed in COCO)
            continue

        lines = lbl_path.read_text().strip().splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            bbox = yolo_to_coco_bbox(line, w, h)
            if bbox is None:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            ann_id += 1

    coco = {
        "info": {"description": "LLVIP test GT (from YOLO labels)"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }

    OUT_JSON.write_text(json.dumps(coco, indent=2))
    print(f"Saved COCO GT to: {OUT_JSON}")
    print(f"Images: {len(images)} | Annotations: {len(annotations)}")

if __name__ == "__main__":
    main()
