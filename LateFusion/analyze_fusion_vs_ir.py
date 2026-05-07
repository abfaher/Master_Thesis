import os
import json
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Any

from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO

from Utils.utils import xywh_to_xyxy, iou_xyxy


def load_predictions_by_image(pred_json_path: str) -> Dict[int, List[Dict[str, Any]]]:
    with open(pred_json_path, "r") as f:
        preds = json.load(f)

    by_image = defaultdict(list)
    for p in preds:
        by_image[p["image_id"]].append(p)
    return by_image


def load_gt_by_image(coco_gt: COCO) -> Dict[int, List[Dict[str, Any]]]:
    gt_by_image = defaultdict(list)
    ann_ids = coco_gt.getAnnIds()
    anns = coco_gt.loadAnns(ann_ids)
    for ann in anns:
        gt_by_image[ann["image_id"]].append(ann)
    return gt_by_image


def match_predictions_to_gt(
    gt_boxes: List[Dict[str, Any]],
    pred_boxes: List[Dict[str, Any]],
    iou_thr: float = 0.5,
    score_thr: float = 0.25,
) -> Dict[str, Any]:
    """
    Greedy one-to-one matching:
    - filter predictions by score
    - sort by descending confidence
    - each prediction matches the best unmatched GT if IoU >= threshold
    """
    filtered_preds = [p for p in pred_boxes if p.get("score", 1.0) >= score_thr]
    filtered_preds = sorted(filtered_preds, key=lambda x: x.get("score", 1.0), reverse=True)

    gt_xyxy = [xywh_to_xyxy(g["bbox"]) for g in gt_boxes]
    pred_xyxy = [xywh_to_xyxy(p["bbox"]) for p in filtered_preds]

    matched_gt = set()
    matched_pred = set()
    pred_match_info = []

    for p_idx, p_box in enumerate(pred_xyxy):
        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, g_box in enumerate(gt_xyxy):
            if g_idx in matched_gt:
                continue
            iou_val = iou_xyxy(p_box, g_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = g_idx

        if best_gt_idx >= 0 and best_iou >= iou_thr:
            matched_gt.add(best_gt_idx)
            matched_pred.add(p_idx)
            pred_match_info.append({
                "pred_index": p_idx,
                "matched": True,
                "gt_index": best_gt_idx,
                "iou": best_iou,
                "score": filtered_preds[p_idx].get("score", 1.0),
                "bbox": filtered_preds[p_idx]["bbox"],
            })
        else:
            pred_match_info.append({
                "pred_index": p_idx,
                "matched": False,
                "gt_index": None,
                "iou": best_iou,
                "score": filtered_preds[p_idx].get("score", 1.0),
                "bbox": filtered_preds[p_idx]["bbox"],
            })

    tp = len(matched_pred)
    fp = len(filtered_preds) - tp
    fn = len(gt_boxes) - len(matched_gt)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_gt_indices": matched_gt,
        "matched_pred_indices": matched_pred,
        "pred_match_info": pred_match_info,
        "filtered_preds": filtered_preds,
    }


def classify_fusion_vs_ir(ir_stats: Dict[str, Any], fusion_stats: Dict[str, Any]) -> str:
    """
    Main supervisor-oriented criterion:
    fusion is better if it reduces FN, i.e. improves recall locally.
    """
    if fusion_stats["fn"] < ir_stats["fn"]:
        return "better"
    if fusion_stats["fn"] > ir_stats["fn"]:
        return "worse"
    return "same"


def draw_boxes(
    image: Image.Image,
    gt_boxes: List[Dict[str, Any]],
    pred_match_info: List[Dict[str, Any]],
    title: str,
    gt_color=(0, 255, 0),
    tp_color=(0, 128, 255),
    fp_color=(255, 0, 0),
) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # GT in green
    for gt in gt_boxes:
        x1, y1, x2, y2 = xywh_to_xyxy(gt["bbox"])
        draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=3)

    # Predictions: TP in blue, FP in red
    for p in pred_match_info:
        x1, y1, x2, y2 = xywh_to_xyxy(p["bbox"])
        color = tp_color if p["matched"] else fp_color
        label = f"{'TP' if p['matched'] else 'FP'} {p['score']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, max(0, y1 - 18)), label, fill=color, font=font)

    draw.text((10, 10), title, fill=(255, 255, 0), font=font)
    return img


def make_side_by_side(img_left: Image.Image, img_right: Image.Image,
                      caption_left: str, caption_right: str) -> Image.Image:
    pad = 20
    header_h = 40
    width = img_left.width + img_right.width + pad
    height = max(img_left.height, img_right.height) + header_h

    canvas = Image.new("RGB", (width, height), (25, 25, 25))
    canvas.paste(img_left, (0, header_h))
    canvas.paste(img_right, (img_left.width + pad, header_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    draw.text((10, 8), caption_left, fill=(255, 255, 255), font=font)
    draw.text((img_left.width + pad + 10, 8), caption_right, fill=(255, 255, 255), font=font)
    return canvas


def resolve_image_path(image_root: str, file_name: str) -> str:
    """
    In LLVIP COCO json, file_name may already be basename or partial path.
    Since we now pass infrared/test/images directly, we first try basename.
    """
    base = os.path.basename(file_name)
    candidate = os.path.join(image_root, base)
    if os.path.exists(candidate):
        return candidate

    candidate2 = os.path.join(image_root, file_name)
    if os.path.exists(candidate2):
        return candidate2

    return candidate


def main():
    parser = argparse.ArgumentParser(description="Per-image IR vs Fusion analysis on LLVIP")
    parser.add_argument("--gt", required=True, help="COCO GT json")
    parser.add_argument("--ir", required=True, help="IR predictions json")
    parser.add_argument("--fusion", required=True, help="Fusion predictions json")
    parser.add_argument("--image-root", required=True, help="Usually ../LLVIP/infrared/test/images")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--score", type=float, default=0.25, help="Score threshold")
    parser.add_argument("--top-k-vis", type=int, default=30, help="Number of better images to visualize")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    coco_gt = COCO(args.gt)
    gt_by_image = load_gt_by_image(coco_gt)
    ir_by_image = load_predictions_by_image(args.ir)
    fusion_by_image = load_predictions_by_image(args.fusion)

    image_ids = sorted(coco_gt.getImgIds())

    summary_rows = []
    better_images = []
    worse_images = []
    same_images = []

    totals = {
        "ir_tp": 0, "ir_fp": 0, "ir_fn": 0,
        "fusion_tp": 0, "fusion_fp": 0, "fusion_fn": 0,
    }

    for image_id in image_ids:
        img_info = coco_gt.loadImgs([image_id])[0]
        file_name = img_info["file_name"]

        gt_boxes = gt_by_image.get(image_id, [])
        ir_preds = ir_by_image.get(image_id, [])
        fusion_preds = fusion_by_image.get(image_id, [])

        ir_stats = match_predictions_to_gt(gt_boxes, ir_preds, args.iou, args.score)
        fusion_stats = match_predictions_to_gt(gt_boxes, fusion_preds, args.iou, args.score)

        status = classify_fusion_vs_ir(ir_stats, fusion_stats)

        row = {
            "image_id": image_id,
            "file_name": file_name,
            "num_gt": len(gt_boxes),

            "ir_tp": ir_stats["tp"],
            "ir_fp": ir_stats["fp"],
            "ir_fn": ir_stats["fn"],

            "fusion_tp": fusion_stats["tp"],
            "fusion_fp": fusion_stats["fp"],
            "fusion_fn": fusion_stats["fn"],

            "delta_tp": fusion_stats["tp"] - ir_stats["tp"],
            "delta_fp": fusion_stats["fp"] - ir_stats["fp"],
            "delta_fn": fusion_stats["fn"] - ir_stats["fn"],

            "fusion_better_recall": status,
        }
        summary_rows.append(row)

        totals["ir_tp"] += ir_stats["tp"]
        totals["ir_fp"] += ir_stats["fp"]
        totals["ir_fn"] += ir_stats["fn"]
        totals["fusion_tp"] += fusion_stats["tp"]
        totals["fusion_fp"] += fusion_stats["fp"]
        totals["fusion_fn"] += fusion_stats["fn"]

        entry = (row, ir_stats, fusion_stats)
        if status == "better":
            better_images.append(entry)
        elif status == "worse":
            worse_images.append(entry)
        else:
            same_images.append(entry)

    csv_path = os.path.join(args.out_dir, "per_image_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = {
        "num_images": len(image_ids),
        "totals": totals,
        "num_better_recall": len(better_images),
        "num_worse_recall": len(worse_images),
        "num_same_recall": len(same_images),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2)

    # Sort better images:
    # more FN reduction first, then smaller FP increase
    better_images_sorted = sorted(
        better_images,
        key=lambda x: (x[0]["delta_fn"], x[0]["delta_fp"])
    )

    num_to_vis = min(args.top_k_vis, len(better_images_sorted))
    for idx in range(num_to_vis):
        row, ir_stats, fusion_stats = better_images_sorted[idx]
        image_id = row["image_id"]
        file_name = row["file_name"]

        img_path = resolve_image_path(args.image_root, file_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Could not find image for {file_name} -> tried {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        gt_boxes = gt_by_image.get(image_id, [])

        ir_panel = draw_boxes(
            image=image,
            gt_boxes=gt_boxes,
            pred_match_info=ir_stats["pred_match_info"],
            title=f"IR-only | TP={row['ir_tp']} FP={row['ir_fp']} FN={row['ir_fn']}"
        )
        fusion_panel = draw_boxes(
            image=image,
            gt_boxes=gt_boxes,
            pred_match_info=fusion_stats["pred_match_info"],
            title=f"Fusion | TP={row['fusion_tp']} FP={row['fusion_fp']} FN={row['fusion_fn']}"
        )

        combined = make_side_by_side(ir_panel, fusion_panel, "IR-only", "Fusion")

        out_name = f"{idx:03d}_img{image_id}_dfn{row['delta_fn']}_dfp{row['delta_fp']}.png"
        combined.save(os.path.join(vis_dir, out_name))

    print("\nDone.")
    print(f"CSV saved to: {csv_path}")
    print(f"Visualizations saved to: {vis_dir}")
    print("\nSummary:")
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()