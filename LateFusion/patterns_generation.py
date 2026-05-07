#!/usr/bin/env python3
"""
Mine clear visual comparison patterns between IR-only and Fusion detections.

Patterns extracted:
1. crowded_separation     -> fusion handles crowded scenes better
2. recall_gain            -> fusion detects GT missed by IR
3. fp_reduction           -> fusion removes false positives made by IR
4. better_localization    -> fusion localizes same GT better than IR
5. stronger_confidence    -> fusion detects same GT with much higher confidence

Expected inputs:
- COCO-style ground-truth annotations JSON
- COCO-style IR-only predictions JSON
- COCO-style Fusion predictions JSON
- image root directory matching "file_name" in GT JSON

Outputs:
- output_dir/
    crowded_separation/
    recall_gain/
    fp_reduction/
    better_localization/
    stronger_confidence/
    pattern_summary.csv
    all_candidates.csv

Usage example:
python mine_patterns.py \
    --gt /workspace/LLVIP/Annotations/test.json \
    --ir_pred /workspace/.../ir_predictions.json \
    --fusion_pred /workspace/.../fusion_predictions.json \
    --image_root /workspace/LLVIP/infrared/test/images \
    --output_dir /workspace/EarlyFusion/pattern_mining \
    --topk 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class GTBox:
    image_id: int
    box_xyxy: Tuple[float, float, float, float]
    category_id: int


@dataclass
class PredBox:
    image_id: int
    box_xyxy: Tuple[float, float, float, float]
    score: float
    category_id: int


@dataclass
class Match:
    gt_idx: int
    pred_idx: int
    iou: float


@dataclass
class Candidate:
    pattern: str
    image_id: int
    file_name: str
    score: float
    note: str
    ir_tp: int
    ir_fp: int
    ir_fn: int
    fusion_tp: int
    fusion_fp: int
    fusion_fn: int
    num_gt: int


# -----------------------------
# Utilities
# -----------------------------

def coco_xywh_to_xyxy(box: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return (x, y, x + w, y + h)


def iou_xyxy(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def greedy_match(
    gts: List[GTBox],
    preds: List[PredBox],
    iou_thr: float = 0.5,
) -> Tuple[List[Match], List[int], List[int]]:
    """
    Greedy one-to-one matching by descending prediction score.
    Returns:
      matches,
      unmatched_gt_indices,
      unmatched_pred_indices
    """
    preds_sorted_idx = sorted(range(len(preds)), key=lambda i: preds[i].score, reverse=True)
    matched_gts = set()
    matched_preds = set()
    matches: List[Match] = []

    for pidx in preds_sorted_idx:
        pred = preds[pidx]
        best_gt = None
        best_iou = 0.0

        for gidx, gt in enumerate(gts):
            if gidx in matched_gts:
                continue
            if gt.category_id != pred.category_id:
                continue
            iou = iou_xyxy(gt.box_xyxy, pred.box_xyxy)
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_gt = gidx

        if best_gt is not None:
            matched_gts.add(best_gt)
            matched_preds.add(pidx)
            matches.append(Match(gt_idx=best_gt, pred_idx=pidx, iou=best_iou))

    unmatched_gts = [i for i in range(len(gts)) if i not in matched_gts]
    unmatched_preds = [i for i in range(len(preds)) if i not in matched_preds]
    return matches, unmatched_gts, unmatched_preds


def load_coco_gt(gt_json_path: str, category_filter: Optional[int] = None):
    with open(gt_json_path, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    gt_by_image: Dict[int, List[GTBox]] = {img_id: [] for img_id in images.keys()}

    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        if category_filter is not None and cat_id != category_filter:
            continue
        img_id = ann["image_id"]
        gt_by_image.setdefault(img_id, []).append(
            GTBox(
                image_id=img_id,
                box_xyxy=coco_xywh_to_xyxy(ann["bbox"]),
                category_id=cat_id,
            )
        )

    return images, gt_by_image


def load_coco_predictions(pred_json_path: str,
                          score_thr: float = 0.001,
                          category_filter: Optional[int] = None):
    with open(pred_json_path, "r") as f:
        preds = json.load(f)

    pred_by_image: Dict[int, List[PredBox]] = {}
    for p in preds:
        score = float(p.get("score", 0.0))
        cat_id = p["category_id"]
        if score < score_thr:
            continue
        if category_filter is not None and cat_id != category_filter:
            continue
        img_id = p["image_id"]
        pred_by_image.setdefault(img_id, []).append(
            PredBox(
                image_id=img_id,
                box_xyxy=coco_xywh_to_xyxy(p["bbox"]),
                score=score,
                category_id=cat_id,
            )
        )
    return pred_by_image


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def draw_boxes(
    image: np.ndarray,
    gt_boxes: List[GTBox],
    preds: List[PredBox],
    matches: List[Match],
    title: str,
    show_scores: bool = True,
) -> np.ndarray:
    """
    Draw GT in green, matched predictions in blue, unmatched predictions in red.
    """
    canvas = image.copy()

    # GT: green
    for gt in gt_boxes:
        x1, y1, x2, y2 = map(int, gt.box_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    matched_pred_indices = {m.pred_idx for m in matches}

    # matched preds: blue
    for idx, pred in enumerate(preds):
        x1, y1, x2, y2 = map(int, pred.box_xyxy)
        if idx in matched_pred_indices:
            color = (255, 180, 0)  # blue-ish in BGR
        else:
            color = (0, 0, 255)    # red
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if show_scores:
            cv2.putText(
                canvas,
                f"{pred.score:.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    # top banner
    h, w = canvas.shape[:2]
    banner_h = 28
    out = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
    out[banner_h:] = canvas
    cv2.putText(
        out,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def make_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    w1 = left.shape[1]
    w2 = right.shape[1]

    if left.shape[0] != h:
        pad = np.zeros((h - left.shape[0], w1, 3), dtype=np.uint8)
        left = np.vstack([left, pad])
    if right.shape[0] != h:
        pad = np.zeros((h - right.shape[0], w2, 3), dtype=np.uint8)
        right = np.vstack([right, pad])

    sep = np.zeros((h, 8, 3), dtype=np.uint8)
    return np.hstack([left, sep, right])


# -----------------------------
# Pattern mining logic
# -----------------------------

def summarize_counts(num_gt: int, matches: List[Match], unmatched_preds: List[int]) -> Tuple[int, int, int]:
    tp = len(matches)
    fp = len(unmatched_preds)
    fn = num_gt - tp
    return tp, fp, fn


def build_gt_to_match_map(matches: List[Match]) -> Dict[int, Match]:
    return {m.gt_idx: m for m in matches}


def mean_pred_score(preds: List[PredBox], matches: List[Match]) -> float:
    if not matches:
        return 0.0
    return float(np.mean([preds[m.pred_idx].score for m in matches]))


def mine_patterns(
    images: Dict[int, dict],
    gt_by_image: Dict[int, List[GTBox]],
    ir_by_image: Dict[int, List[PredBox]],
    fusion_by_image: Dict[int, List[PredBox]],
    iou_thr: float,
    crowded_gt_min: int,
    localization_iou_gain: float,
    confidence_gain: float,
) -> Tuple[List[Candidate], Dict[int, dict]]:
    """
    Returns:
      candidates list
      per-image analysis dict for later visualization
    """
    candidates: List[Candidate] = []
    analysis: Dict[int, dict] = {}

    for image_id, img_info in images.items():
        gt_boxes = gt_by_image.get(image_id, [])
        ir_preds = ir_by_image.get(image_id, [])
        fusion_preds = fusion_by_image.get(image_id, [])

        ir_matches, ir_unmatched_gt, ir_unmatched_preds = greedy_match(gt_boxes, ir_preds, iou_thr)
        fusion_matches, fusion_unmatched_gt, fusion_unmatched_preds = greedy_match(gt_boxes, fusion_preds, iou_thr)

        ir_tp, ir_fp, ir_fn = summarize_counts(len(gt_boxes), ir_matches, ir_unmatched_preds)
        fusion_tp, fusion_fp, fusion_fn = summarize_counts(len(gt_boxes), fusion_matches, fusion_unmatched_preds)

        analysis[image_id] = {
            "gt_boxes": gt_boxes,
            "ir_preds": ir_preds,
            "fusion_preds": fusion_preds,
            "ir_matches": ir_matches,
            "fusion_matches": fusion_matches,
            "ir_unmatched_gt": ir_unmatched_gt,
            "fusion_unmatched_gt": fusion_unmatched_gt,
            "ir_unmatched_preds": ir_unmatched_preds,
            "fusion_unmatched_preds": fusion_unmatched_preds,
            "ir_tp": ir_tp,
            "ir_fp": ir_fp,
            "ir_fn": ir_fn,
            "fusion_tp": fusion_tp,
            "fusion_fp": fusion_fp,
            "fusion_fn": fusion_fn,
            "num_gt": len(gt_boxes),
        }

        if len(gt_boxes) == 0:
            continue

        file_name = img_info["file_name"]

        # -----------------------------
        # Pattern 1: crowded separation
        # -----------------------------
        if len(gt_boxes) >= crowded_gt_min:
            # fusion must improve TP or FN in crowded scenes
            crowded_gain = (fusion_tp - ir_tp) + (ir_fn - fusion_fn)
            if crowded_gain > 0:
                score = 3.0 * crowded_gain + 0.2 * len(gt_boxes) - 0.5 * max(0, fusion_fp - ir_fp)
                candidates.append(Candidate(
                    pattern="crowded_separation",
                    image_id=image_id,
                    file_name=file_name,
                    score=score,
                    note=f"Crowded scene with {len(gt_boxes)} GTs. Fusion TP={fusion_tp}, IR TP={ir_tp}.",
                    ir_tp=ir_tp, ir_fp=ir_fp, ir_fn=ir_fn,
                    fusion_tp=fusion_tp, fusion_fp=fusion_fp, fusion_fn=fusion_fn,
                    num_gt=len(gt_boxes),
                ))

        # -----------------------------
        # Pattern 2: recall gain
        # GT detected by fusion but missed by IR
        # -----------------------------
        ir_gt_map = build_gt_to_match_map(ir_matches)
        fusion_gt_map = build_gt_to_match_map(fusion_matches)

        recall_gain_count = 0
        recall_gain_score = 0.0
        for gidx in range(len(gt_boxes)):
            ir_hit = gidx in ir_gt_map
            fusion_hit = gidx in fusion_gt_map
            if (not ir_hit) and fusion_hit:
                recall_gain_count += 1
                recall_gain_score += fusion_preds[fusion_gt_map[gidx].pred_idx].score

        if recall_gain_count > 0:
            score = 5.0 * recall_gain_count + recall_gain_score
            candidates.append(Candidate(
                pattern="recall_gain",
                image_id=image_id,
                file_name=file_name,
                score=score,
                note=f"Fusion recovered {recall_gain_count} GT(s) missed by IR.",
                ir_tp=ir_tp, ir_fp=ir_fp, ir_fn=ir_fn,
                fusion_tp=fusion_tp, fusion_fp=fusion_fp, fusion_fn=fusion_fn,
                num_gt=len(gt_boxes),
            ))

        # -----------------------------
        # Pattern 3: false positive reduction
        # IR makes FP that fusion removes
        # -----------------------------
        ir_fp_removed = max(0, ir_fp - fusion_fp)
        if ir_fp_removed > 0 and fusion_tp >= max(ir_tp - 1, 0):
            score = 4.0 * ir_fp_removed + 0.5 * fusion_tp
            candidates.append(Candidate(
                pattern="fp_reduction",
                image_id=image_id,
                file_name=file_name,
                score=score,
                note=f"Fusion reduced FP count from {ir_fp} to {fusion_fp}.",
                ir_tp=ir_tp, ir_fp=ir_fp, ir_fn=ir_fn,
                fusion_tp=fusion_tp, fusion_fp=fusion_fp, fusion_fn=fusion_fn,
                num_gt=len(gt_boxes),
            ))

        # -----------------------------
        # Pattern 4: better localization
        # same GT detected by both, but fusion IoU higher
        # -----------------------------
        loc_gain_count = 0
        loc_gain_total = 0.0
        for gidx in range(len(gt_boxes)):
            if gidx in ir_gt_map and gidx in fusion_gt_map:
                ir_iou = ir_gt_map[gidx].iou
                fusion_iou = fusion_gt_map[gidx].iou
                gain = fusion_iou - ir_iou
                if gain >= localization_iou_gain:
                    loc_gain_count += 1
                    loc_gain_total += gain

        if loc_gain_count > 0:
            score = 3.0 * loc_gain_count + 10.0 * loc_gain_total
            candidates.append(Candidate(
                pattern="better_localization",
                image_id=image_id,
                file_name=file_name,
                score=score,
                note=f"Fusion improved IoU on {loc_gain_count} GT(s).",
                ir_tp=ir_tp, ir_fp=ir_fp, ir_fn=ir_fn,
                fusion_tp=fusion_tp, fusion_fp=fusion_fp, fusion_fn=fusion_fn,
                num_gt=len(gt_boxes),
            ))

        # -----------------------------
        # Pattern 5: stronger confidence
        # same GT detected by both, fusion confidence much higher
        # -----------------------------
        conf_gain_count = 0
        conf_gain_total = 0.0
        for gidx in range(len(gt_boxes)):
            if gidx in ir_gt_map and gidx in fusion_gt_map:
                ir_score = ir_preds[ir_gt_map[gidx].pred_idx].score
                fusion_score = fusion_preds[fusion_gt_map[gidx].pred_idx].score
                gain = fusion_score - ir_score
                if gain >= confidence_gain:
                    conf_gain_count += 1
                    conf_gain_total += gain

        if conf_gain_count > 0:
            score = 2.0 * conf_gain_count + 5.0 * conf_gain_total
            candidates.append(Candidate(
                pattern="stronger_confidence",
                image_id=image_id,
                file_name=file_name,
                score=score,
                note=f"Fusion confidence improved on {conf_gain_count} GT(s).",
                ir_tp=ir_tp, ir_fp=ir_fp, ir_fn=ir_fn,
                fusion_tp=fusion_tp, fusion_fp=fusion_fp, fusion_fn=fusion_fn,
                num_gt=len(gt_boxes),
            ))

    return candidates, analysis


# -----------------------------
# Visualization + export
# -----------------------------

def save_candidate_visualization(
    cand: Candidate,
    images: Dict[int, dict],
    analysis: Dict[int, dict],
    image_root: str,
    out_path: str,
):
    img_info = images[cand.image_id]
    img_path = os.path.join(image_root, img_info["file_name"])

    im = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    a = analysis[cand.image_id]

    left_title = f"IR-only | TP={a['ir_tp']} FP={a['ir_fp']} FN={a['ir_fn']}"
    right_title = f"Fusion | TP={a['fusion_tp']} FP={a['fusion_fp']} FN={a['fusion_fn']}"

    left = draw_boxes(
        im,
        a["gt_boxes"],
        a["ir_preds"],
        a["ir_matches"],
        left_title,
        show_scores=True,
    )
    right = draw_boxes(
        im,
        a["gt_boxes"],
        a["fusion_preds"],
        a["fusion_matches"],
        right_title,
        show_scores=True,
    )

    side = make_side_by_side(left, right)

    # footer text
    footer_h = 54
    footer = np.zeros((footer_h, side.shape[1], 3), dtype=np.uint8)
    text1 = f"Pattern: {cand.pattern}"
    text2 = cand.note
    cv2.putText(footer, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(footer, text2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    final = np.vstack([side, footer])
    cv2.imwrite(out_path, final)


def write_csv(path: str, rows: List[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------
# Main
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=str, required=True, help="COCO GT annotations JSON")
    p.add_argument("--ir_pred", type=str, required=True, help="IR-only predictions JSON")
    p.add_argument("--fusion_pred", type=str, required=True, help="Fusion predictions JSON")
    p.add_argument("--image_root", type=str, required=True, help="Directory where COCO file_name images are stored")
    p.add_argument("--output_dir", type=str, required=True, help="Output folder")
    p.add_argument("--category_id", type=int, default=1, help="Pedestrian class category id")
    p.add_argument("--score_thr", type=float, default=0.001, help="Prediction score threshold")
    p.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for TP matching")
    p.add_argument("--topk", type=int, default=8, help="How many examples to save per pattern")
    p.add_argument("--crowded_gt_min", type=int, default=4, help="Minimum GT count to qualify as crowded")
    p.add_argument("--localization_iou_gain", type=float, default=0.10, help="Min IoU improvement for better localization")
    p.add_argument("--confidence_gain", type=float, default=0.20, help="Min score improvement for stronger confidence")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    print("[INFO] Loading GT...")
    images, gt_by_image = load_coco_gt(args.gt, category_filter=args.category_id)

    print("[INFO] Loading predictions...")
    ir_by_image = load_coco_predictions(args.ir_pred, score_thr=args.score_thr, category_filter=args.category_id)
    fusion_by_image = load_coco_predictions(args.fusion_pred, score_thr=args.score_thr, category_filter=args.category_id)

    print("[INFO] Mining patterns...")
    candidates, analysis = mine_patterns(
        images=images,
        gt_by_image=gt_by_image,
        ir_by_image=ir_by_image,
        fusion_by_image=fusion_by_image,
        iou_thr=args.iou_thr,
        crowded_gt_min=args.crowded_gt_min,
        localization_iou_gain=args.localization_iou_gain,
        confidence_gain=args.confidence_gain,
    )

    # Save all candidates
    all_rows = [asdict(c) for c in sorted(candidates, key=lambda x: (x.pattern, -x.score))]
    write_csv(os.path.join(args.output_dir, "all_candidates.csv"), all_rows)

    patterns = [
        "crowded_separation",
        "recall_gain",
        "fp_reduction",
        "better_localization",
        "stronger_confidence",
    ]

    summary_rows = []

    for pattern in patterns:
        pattern_dir = os.path.join(args.output_dir, pattern)
        ensure_dir(pattern_dir)

        pattern_candidates = [c for c in candidates if c.pattern == pattern]
        pattern_candidates.sort(key=lambda x: x.score, reverse=True)

        # keep only top-k unique image_ids
        top_unique: List[Candidate] = []
        seen = set()
        for c in pattern_candidates:
            if c.image_id in seen:
                continue
            seen.add(c.image_id)
            top_unique.append(c)
            if len(top_unique) >= args.topk:
                break

        print(f"[INFO] {pattern}: saving {len(top_unique)} examples")

        for rank, cand in enumerate(top_unique, start=1):
            out_name = f"{rank:02d}_img{cand.image_id}.jpg"
            out_path = os.path.join(pattern_dir, out_name)
            save_candidate_visualization(
                cand=cand,
                images=images,
                analysis=analysis,
                image_root=args.image_root,
                out_path=out_path,
            )
            summary_rows.append({
                "pattern": cand.pattern,
                "rank": rank,
                "image_id": cand.image_id,
                "file_name": cand.file_name,
                "score": round(cand.score, 4),
                "note": cand.note,
                "ir_tp": cand.ir_tp,
                "ir_fp": cand.ir_fp,
                "ir_fn": cand.ir_fn,
                "fusion_tp": cand.fusion_tp,
                "fusion_fp": cand.fusion_fp,
                "fusion_fn": cand.fusion_fn,
                "num_gt": cand.num_gt,
                "saved_path": out_path,
            })

    write_csv(os.path.join(args.output_dir, "pattern_summary.csv"), summary_rows)
    print(f"[DONE] Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()