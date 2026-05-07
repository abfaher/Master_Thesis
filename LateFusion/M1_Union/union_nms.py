import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict

# Make Utils/ importable since it's not a package
ROOT = Path(__file__).resolve().parents[1]   # LateFusion/
UTILS_DIR = ROOT/"Utils"
sys.path.append(str(UTILS_DIR))

from utils import xywh_to_xyxy, iou_xyxy

IR_JSON  = Path("/workspace/YOLOv8_IR_only/runs/detect/val/predictions.json")
RGB_JSON = Path("/workspace/YOLOv8_RGB_only/runs/detect/val/predictions.json")
OUT_JSON = Path(__file__).resolve().parent/"predictions_fused_union_nms.json"

IOU_THR = 0.60  # NMS threshold
SCORE_THR = 0.01

def nms(dets, iou_thr=0.6):
    """
    Compute a classical Non-Maximum Suppresion.
    Idea: If multiple boxes overlap heavily and likely represent the same person,
    keep only the best one and discard the duplicates.

    iou_thr=0.6 meaning if the two boxes overlap with IoU > 0.6, they are considered duplicates -> suppress the weaker one.

    dets: list of dict with fields bbox(xywh), score, where each dict represents one detected person in one image.
    returns filtered list (same dict objects)
    """
    dets = [d for d in dets if d["score"] >= SCORE_THR]
    dets.sort(key=lambda d: d["score"], reverse=True)

    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        best_xyxy = xywh_to_xyxy(best["bbox"])

        remaining = []
        for d in dets:
            if iou_xyxy(best_xyxy, xywh_to_xyxy(d["bbox"])) <= iou_thr:  # probably a different person
                remaining.append(d)  # keep it
        dets = remaining
    return keep

def load_json(p):
    return json.loads(p.read_text())

def main():
    assert IR_JSON.exists(), f"Missing: {IR_JSON}"
    assert RGB_JSON.exists(), f"Missing: {RGB_JSON}"

    ir = load_json(IR_JSON)
    rgb = load_json(RGB_JSON)

    # group by image_id
    by_img = defaultdict(list)
    for d in ir:
        by_img[d["image_id"]].append(d)
    for d in rgb:
        by_img[d["image_id"]].append(d)

    fused = []
    for img_id, dets in by_img.items():
        # (optional) you can also enforce same category_id=1
        dets = [d for d in dets if int(d.get("category_id", 1)) == 1]
        kept = nms(dets, IOU_THR)
        fused.extend(kept)

    OUT_JSON.write_text(json.dumps(fused, indent=2))
    print(f"Saved fused predictions to: {OUT_JSON}")
    print(f"IR dets: {len(ir)} | RGB dets: {len(rgb)} | Fused dets: {len(fused)}")
    print(f"Used IOU_THR={IOU_THR} SCORE_THR={SCORE_THR}")

if __name__ == "__main__":
    main()
    