import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict

# Make Utils/ importable (since it's not a package)
ROOT = Path(__file__).resolve().parents[1]  # LateFusion/
UTILS_DIR = ROOT / "Utils"
sys.path.append(str(UTILS_DIR))

from utils import xywh_to_xyxy, iou_xyxy, xyxy_to_xywh, clamp_box_xyxy


def weighted_average_box(cluster, model_weights):
    """
    Fuse a cluster of overlapping detections into ONE box (WBF).

    cluster: list of det dicts. Each det has:
      - bbox: [x, y, w, h] in COCO xywh (pixels)
      - score
      - _src: "ir" or "rgb" (we add it)
    model_weights: dict {"ir": 1.0, "rgb": 0.7} etc.

    Coordinate weights:
      w_i = score_i * model_weight(src_i)
    """
    ws = []
    xs1, ys1, xs2, ys2 = [], [], [], []

    for d in cluster:
        src = d.get("_src", "rgb")
        mw = float(model_weights.get(src, 1.0))
        s = float(d["score"])
        w = s * mw

        x1, y1, x2, y2 = xywh_to_xyxy(d["bbox"])
        xs1.append(x1)
        ys1.append(y1)
        xs2.append(x2)
        ys2.append(y2)

        ws.append(w)

    wsum = sum(ws) if sum(ws) > 0 else 1.0

    fx1 = sum(w * x for w, x in zip(ws, xs1)) / wsum
    fy1 = sum(w * y for w, y in zip(ws, ys1)) / wsum
    fx2 = sum(w * x for w, x in zip(ws, xs2)) / wsum
    fy2 = sum(w * y for w, y in zip(ws, ys2)) / wsum

    fused_xyxy = clamp_box_xyxy([fx1, fy1, fx2, fy2])

    # Fused score = sum(score_i x model_weight) / sum(model_weight)
    num, den = 0.0, 0.0
    for d in cluster:
        src = d.get("_src", "rgb")
        mw = float(model_weights.get(src, 1.0))
        s = float(d["score"])
        num += mw * s
        den += mw
    fused_score = num / den if den > 0 else float(max(d["score"] for d in cluster))

    return fused_xyxy, float(fused_score)


def wbf_fuse(dets, iou_thr, score_thr, model_weights):
    """
    WBF loop:
    - pick best remaining box
    - gather all overlapping boxes (IoU >= threshold)
    - merge them into one box (weighted average)
    - repeat
    """
    dets = [d for d in dets if float(d.get("score", 0.0)) >= score_thr]
    dets.sort(key=lambda d: float(d["score"]), reverse=True)  # detections are sorted from highest  confidence to lowest 

    fused_out = []

    while dets:
        seed = dets.pop(0)
        cluster = [seed]  # list of boxes representing the same person

        # "reference" box to grow the cluster
        ref_xyxy = xywh_to_xyxy(seed["bbox"])

        remaining = []
        for d in dets:
            d_xyxy = xywh_to_xyxy(d["bbox"])
            if iou_xyxy(ref_xyxy, d_xyxy) >= iou_thr:
                cluster.append(d)
                # Update ref box to the current fused box
                ref_xyxy, _ = weighted_average_box(cluster, model_weights)
            else:
                remaining.append(d)

        dets = remaining

        fused_xyxy, fused_score = weighted_average_box(cluster, model_weights)

        fused_out.append({
            "image_id": int(seed["image_id"]),
            "category_id": int(seed.get("category_id", 1)),
            "bbox": xyxy_to_xywh(fused_xyxy),
            "score": float(fused_score),
        })

    return fused_out


def load_json(p: Path):
    return json.loads(p.read_text())


def main():
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", type=str, default="/workspace/YOLOv8_IR_only/runs/detect/val/predictions.json")
    parser.add_argument("--rgb", type=str, default="/workspace/YOLOv8_RGB_only/runs/detect/val/predictions.json")
    parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "predictions_fused_wbf.json"))

    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--score", type=float, default=0.01)

    parser.add_argument("--w_ir", type=float, default=1.0)
    parser.add_argument("--w_rgb", type=float, default=0.4)

    args = parser.parse_args()

    ir_json = Path(args.ir)
    rgb_json = Path(args.rgb)
    out_json = Path(args.out)

    assert ir_json.exists(), f"Missing: {ir_json}"
    assert rgb_json.exists(), f"Missing: {rgb_json}"

    ir = load_json(ir_json)
    rgb = load_json(rgb_json)

    model_weights = {"ir": args.w_ir, "rgb": args.w_rgb}

    # Loads IR + RGB predictions
    by_img = defaultdict(list)  # dict assigning each image_id (key) to a list of its detection (value)
    for d in ir:
        if int(d.get("category_id", 1)) != 1:
            continue
        dd = dict(d)
        dd["_src"] = "ir"
        by_img[dd["image_id"]].append(dd)

    for d in rgb:
        if int(d.get("category_id", 1)) != 1:
            continue
        dd = dict(d)
        dd["_src"] = "rgb"
        by_img[dd["image_id"]].append(dd)

    # Run WBF per image
    fused = []
    for _, dets in by_img.items():
        fused.extend(wbf_fuse(dets, args.iou, args.score, model_weights))   

    out_json.write_text(json.dumps(fused, indent=2))
    print(f"Saved fused predictions to: {out_json}")
    print(f"IR dets: {len(ir)} | RGB dets: {len(rgb)} | Fused dets: {len(fused)}")
    print(f"Used IOU_THR={args.iou} SCORE_THR={args.score} MODEL_WEIGHTS={model_weights}")


if __name__ == "__main__":
    main()