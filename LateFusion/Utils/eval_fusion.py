import json
import sys
from pathlib import Path
from collections import defaultdict
import argparse
import io
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]   # LateFusion/
sys.path.append(str(Path(__file__).resolve().parent))  # LateFusion/Utils

from utils import xywh_to_xyxy, iou_xyxy

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

M1_DIR = ROOT/"M1_Union"

GT_JSON = M1_DIR/"llvip_test_gt_coco.json"
IR_JSON  = Path("/workspace/YOLOv8_IR_only/runs/detect/val/predictions.json")
RGB_JSON = Path("/workspace/YOLOv8_RGB_only/runs/detect/val/predictions.json")
FUSED_JSON = M1_DIR/"predictions_fused_union_nms.json"

CONF_THR = 0.25
IOU_THR_PR = 0.50

def load_preds(p):
    preds = json.loads(p.read_text())
    # keep only person category and sanity fields
    out = []
    for d in preds:
        if int(d.get("category_id", 1)) != 1:
            continue
        if "bbox" not in d or "score" not in d or "image_id" not in d:
            continue
        out.append(d)
    return out

def coco_map(gt_json, pred_json):
    """
    Run Coco evaluation and returns key metrics.
    COCOeval stats:
    stats[0] = AP@[0.5:0.95]
    stats[1] = AP@0.5
    stats[6] = AR@[0.5:0.95] (maxDets=100)
    """
    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(pred_json))
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate()
    ev.accumulate()

    # summarize command creates a coco evaluation table, but we silence it
    with redirect_stdout(io.StringIO()):
        ev.summarize()

    return {
        "mAP50_95": float(ev.stats[0]),
        "mAP50": float(ev.stats[1]),
        "AR50_95": float(ev.stats[6]),
    }

def pr_at_threshold(gt_json, preds, conf_thr=0.25, iou_thr=0.5):
    """
    Simple greedy matching per image:
    - filter preds by conf_thr
    - for each image, match highest-score preds to GT boxes if IoU >= iou_thr
    """
    # Gt loading
    gt = json.loads(Path(gt_json).read_text())
    gt_by_img = defaultdict(list)
    for ann in gt["annotations"]:
        gt_by_img[ann["image_id"]].append(xywh_to_xyxy(ann["bbox"]))

    # predictions grouping (using confidence filter)
    pred_by_img = defaultdict(list)
    for d in preds:
        if d["score"] >= conf_thr:
            pred_by_img[d["image_id"]].append(d)

    TP = 0
    FP = 0
    FN = 0

    all_img_ids = set(gt_by_img.keys()) | set(pred_by_img.keys())

    # Matching loop
    for img_id in all_img_ids:
        gt_boxes = gt_by_img.get(img_id, [])
        pr_boxes = pred_by_img.get(img_id, [])

        # sort predictions by score descending
        pr_boxes.sort(key=lambda x: x["score"], reverse=True)

        matched = [False] * len(gt_boxes)
        # for each prediction
        for p in pr_boxes:
            pxy = xywh_to_xyxy(p["bbox"])  # convert its bbox to xyxy
            best_iou = 0.0
            best_j = -1
            # find best IoU among unmatched GT boxes
            for j, gxy in enumerate(gt_boxes):
                if matched[j]:
                    continue
                iou = iou_xyxy(pxy, gxy)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thr and best_j >= 0:
                matched[best_j] = True  # mark gt as matched
                TP += 1
            else:
                FP += 1

        FN += matched.count(False)

    # compute the precision and recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return {"P@conf": precision, "R@conf": recall, "TP": TP, "FP": FP, "FN": FN}

def eval_one(name, pred_path):
    """
    Evaluates one prediction in two ways:
    - Coco mAP
    - Precision/Recall at fixed threshold
    """
    print(f"\n=== {name} ===")
    preds = load_preds(pred_path)

    # For COCOeval, need a json file path, so we write a temp file if needed
    tmp_path = Path(f"_tmp_{name}.json")
    tmp_path.write_text(json.dumps(preds))

    # Coco evalutaion
    m = coco_map(GT_JSON, tmp_path)

    # Simple PR evaluation
    pr = pr_at_threshold(GT_JSON, preds, conf_thr=CONF_THR, iou_thr=IOU_THR_PR)

    # Deletes temp file
    tmp_path.unlink(missing_ok=True)

    print("\n")
    print(f"mAP@0.5:0.95 = {m['mAP50_95']:.4f}")
    print(f"mAP@0.5      = {m['mAP50']:.4f}")
    print(f"P@conf={CONF_THR} IoU={IOU_THR_PR} = {pr['P@conf']:.4f}")
    print(f"R@conf={CONF_THR} IoU={IOU_THR_PR} = {pr['R@conf']:.4f}")
    print(f"TP={pr['TP']} FP={pr['FP']} FN={pr['FN']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fused",
        type=str,
        default=str(FUSED_JSON),
        help="Path to fused predictions JSON"
    )
    args = parser.parse_args()

    fused_path = Path(args.fused)

    assert GT_JSON.exists(), f"Missing GT json: {GT_JSON}"
    assert IR_JSON.exists(), f"Missing IR preds: {IR_JSON}"
    assert RGB_JSON.exists(), f"Missing RGB preds: {RGB_JSON}"
    assert fused_path.exists(), f"Missing fused preds: {fused_path}"

    eval_one("IR-only", IR_JSON)
    eval_one("RGB-only", RGB_JSON)
    eval_one(f"FUSED ({fused_path.name})", fused_path)

if __name__ == "__main__":
    main()
