# import os
# import math
# import json
# import glob
# import cv2
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from dataclasses import dataclass
# from collections import defaultdict

# # ============================================================
# # CONFIG — ADAPTED TO YOUR PROJECT TREE
# # ============================================================

# # Images
# RGB_DIR = "/workspace/datasets/LLVIP/visible/test/images"
# IR_DIR = "/workspace/datasets/LLVIP/infrared/test/images"

# # Ground-truth labels (YOLO txt)
# GT_LABELS_DIR = "/workspace/datasets/LLVIP/infrared/test/labels"

# # Predictions (JSON)
# IR_JSON_PATH = "/workspace/YOLOv8_IR_only/runs/detect/val/predictions.json"
# EARLY_JSON_PATH = "/workspace/EarlyFusion/experiments/earlyfusion_eval2/predictions.json"

# # Output
# OUT_DIR = "/workspace/EarlyFusion/patterns_earlyfusion_vs_ir"

# # Thresholds
# CONF_THRES = 0.25
# IOU_THRES = 0.50

# # Number of examples per pattern
# TOP_K_PER_PATTERN = 6
# TOP_K_OVERALL = 12

# # Visualization
# LINE_THICKNESS = 2
# FONT_SCALE = 0.6
# TARGET_H = 440


# # ============================================================
# # DATA STRUCTURES
# # ============================================================

# @dataclass
# class Box:
#     cls: int
#     x1: float
#     y1: float
#     x2: float
#     y2: float
#     conf: float = 1.0

#     @property
#     def w(self):
#         return max(0.0, self.x2 - self.x1)

#     @property
#     def h(self):
#         return max(0.0, self.y2 - self.y1)

#     @property
#     def area(self):
#         return self.w * self.h

#     @property
#     def cx(self):
#         return 0.5 * (self.x1 + self.x2)

#     @property
#     def cy(self):
#         return 0.5 * (self.y1 + self.y2)


# # ============================================================
# # UTILITIES
# # ============================================================

# def ensure_dir(path):
#     Path(path).mkdir(parents=True, exist_ok=True)

# def find_image_files(root):
#     exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
#     files = []
#     for ext in exts:
#         files.extend(glob.glob(os.path.join(root, ext)))
#     return sorted(files)

# def stem_to_path_dict(root):
#     out = {}
#     for p in find_image_files(root):
#         out[Path(p).stem] = p
#     return out

# def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
#     x1 = (xc - w / 2.0) * img_w
#     y1 = (yc - h / 2.0) * img_h
#     x2 = (xc + w / 2.0) * img_w
#     y2 = (yc + h / 2.0) * img_h
#     return x1, y1, x2, y2

# def parse_yolo_gt(txt_path, img_w, img_h):
#     boxes = []
#     if not os.path.exists(txt_path):
#         return boxes

#     with open(txt_path, "r") as f:
#         lines = [ln.strip() for ln in f.readlines() if ln.strip()]

#     for line in lines:
#         parts = line.split()
#         if len(parts) < 5:
#             continue
#         cls, xc, yc, w, h = parts[:5]
#         cls = int(float(cls))
#         xc, yc, w, h = map(float, [xc, yc, w, h])
#         x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
#         boxes.append(Box(cls=cls, x1=x1, y1=y1, x2=x2, y2=y2, conf=1.0))
#     return boxes

# def load_predictions_json(json_path, conf_thres=0.25):
#     """
#     Expects COCO-like detections:
#     {
#       "image_id": ...,
#       "file_name": "190001.jpg",
#       "category_id": 1,
#       "bbox": [x, y, w, h],
#       "score": 0.89
#     }
#     """
#     with open(json_path, "r") as f:
#         data = json.load(f)

#     by_stem = defaultdict(list)

#     for item in data:
#         score = float(item.get("score", 1.0))
#         if score < conf_thres:
#             continue

#         file_name = item.get("file_name")
#         if file_name is None:
#             # fallback to image_id
#             image_id = str(item["image_id"])
#             stem = image_id
#         else:
#             stem = Path(file_name).stem

#         x, y, w, h = item["bbox"]
#         x1 = float(x)
#         y1 = float(y)
#         x2 = float(x + w)
#         y2 = float(y + h)

#         # category_id in JSON is 1 for person, convert to 0 for consistency
#         cat = int(item.get("category_id", 1))
#         cls = 0 if cat == 1 else cat - 1

#         by_stem[stem].append(Box(cls=cls, x1=x1, y1=y1, x2=x2, y2=y2, conf=score))

#     return by_stem

# def box_iou(a: Box, b: Box):
#     inter_x1 = max(a.x1, b.x1)
#     inter_y1 = max(a.y1, b.y1)
#     inter_x2 = min(a.x2, b.x2)
#     inter_y2 = min(a.y2, b.y2)

#     inter_w = max(0.0, inter_x2 - inter_x1)
#     inter_h = max(0.0, inter_y2 - inter_y1)
#     inter = inter_w * inter_h

#     union = a.area + b.area - inter
#     if union <= 0:
#         return 0.0
#     return inter / union

# def greedy_match(gt_boxes, pred_boxes, iou_thres=0.5):
#     candidates = []
#     for gi, gt in enumerate(gt_boxes):
#         for pi, pr in enumerate(pred_boxes):
#             if gt.cls != pr.cls:
#                 continue
#             iou = box_iou(gt, pr)
#             if iou >= iou_thres:
#                 candidates.append((iou, gi, pi))

#     candidates.sort(reverse=True, key=lambda x: x[0])

#     gt_used = set()
#     pred_used = set()
#     gt_to_pred = {}

#     for iou, gi, pi in candidates:
#         if gi in gt_used or pi in pred_used:
#             continue
#         gt_used.add(gi)
#         pred_used.add(pi)
#         gt_to_pred[gi] = pi

#     unmatched_gt = set(range(len(gt_boxes))) - set(gt_to_pred.keys())
#     unmatched_pred = set(range(len(pred_boxes))) - set(gt_to_pred.values())

#     return gt_to_pred, unmatched_gt, unmatched_pred

# def normalize_ir(ir_img):
#     if len(ir_img.shape) == 3:
#         ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
#     return cv2.normalize(ir_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# def make_fused_visual(rgb, ir_gray):
#     ir_norm = normalize_ir(ir_gray)
#     ir_heat = cv2.applyColorMap(ir_norm, cv2.COLORMAP_JET)
#     fused = cv2.addWeighted(rgb, 0.65, ir_heat, 0.35, 0)
#     return fused

# def draw_boxes(img, boxes, color, label_prefix="", show_conf=True):
#     out = img.copy()
#     for b in boxes:
#         x1, y1, x2, y2 = map(int, [b.x1, b.y1, b.x2, b.y2])
#         cv2.rectangle(out, (x1, y1), (x2, y2), color, LINE_THICKNESS)
#         if show_conf:
#             label = f"{label_prefix}{b.conf:.2f}"
#             cv2.putText(
#                 out, label, (x1, max(15, y1 - 5)),
#                 cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, 2, cv2.LINE_AA
#             )
#     return out

# def draw_highlighted_gt(img, gt_boxes, rescued_gt_indices, default_color=(0, 255, 0), rescued_color=(0, 255, 255)):
#     """
#     GT normal = green
#     GT rescued by Early Fusion = yellow
#     """
#     out = img.copy()
#     for i, b in enumerate(gt_boxes):
#         color = rescued_color if i in rescued_gt_indices else default_color
#         x1, y1, x2, y2 = map(int, [b.x1, b.y1, b.x2, b.y2])
#         cv2.rectangle(out, (x1, y1), (x2, y2), color, LINE_THICKNESS)
#     return out

# def make_panel_title(img, title):
#     out = img.copy()
#     cv2.putText(
#         out, title, (10, 28),
#         cv2.FONT_HERSHEY_SIMPLEX, 0.85,
#         (255, 255, 255), 2, cv2.LINE_AA
#     )
#     return out

# def hstack_resize(images, target_h=440):
#     resized = []
#     for im in images:
#         h, w = im.shape[:2]
#         scale = target_h / float(h)
#         new_w = max(1, int(w * scale))
#         resized.append(cv2.resize(im, (new_w, target_h)))
#     return cv2.hconcat(resized)

# def crop_stats_ir(ir_img, box: Box):
#     h, w = ir_img.shape[:2]
#     x1 = max(0, int(box.x1))
#     y1 = max(0, int(box.y1))
#     x2 = min(w, int(box.x2))
#     y2 = min(h, int(box.y2))
#     if x2 <= x1 or y2 <= y1:
#         return {"ir_mean": 0.0, "ir_std": 0.0}

#     crop = ir_img[y1:y2, x1:x2]
#     return {
#         "ir_mean": float(np.mean(crop)),
#         "ir_std": float(np.std(crop))
#     }

# def nearest_neighbor_distance(target_idx, gt_boxes, img_w, img_h):
#     target = gt_boxes[target_idx]
#     if len(gt_boxes) <= 1:
#         return 1e9
#     dists = []
#     for i, g in enumerate(gt_boxes):
#         if i == target_idx:
#             continue
#         dx = (target.cx - g.cx) / img_w
#         dy = (target.cy - g.cy) / img_h
#         dists.append(math.sqrt(dx * dx + dy * dy))
#     return min(dists) if dists else 1e9

# def count_neighbors(target_idx, gt_boxes, img_w, img_h, radius=0.10):
#     target = gt_boxes[target_idx]
#     c = 0
#     for i, g in enumerate(gt_boxes):
#         if i == target_idx:
#             continue
#         dx = (target.cx - g.cx) / img_w
#         dy = (target.cy - g.cy) / img_h
#         dist = math.sqrt(dx * dx + dy * dy)
#         if dist < radius:
#             c += 1
#     return c

# def assign_pattern(area_ratio, border_flag, ir_std, neighbor_count, nn_dist):
#     if neighbor_count >= 1 or nn_dist < 0.08:
#         return "crowded_scene"
#     if area_ratio < 0.008:
#         return "small_far_targets"
#     if border_flag:
#         return "border_hard_case"
#     if ir_std < 18:
#         return "low_ir_contrast"
#     return "general_complementarity"

# def make_two_panel_figure(
#     rgb, ir, gt_boxes, ir_preds, early_preds, rescued_gt_indices, footer_text=""
# ):
#     ir_vis = cv2.cvtColor(normalize_ir(ir), cv2.COLOR_GRAY2BGR)
#     fused_vis = make_fused_visual(rgb, ir)

#     # left = IR-only
#     ir_panel = draw_highlighted_gt(ir_vis, gt_boxes, rescued_gt_indices)
#     ir_panel = draw_boxes(ir_panel, ir_preds, color=(0, 0, 255), label_prefix="IR ")
#     ir_panel = make_panel_title(ir_panel, "IR-only")

#     # right = Early Fusion
#     early_panel = draw_highlighted_gt(fused_vis, gt_boxes, rescued_gt_indices)
#     early_panel = draw_boxes(early_panel, early_preds, color=(255, 0, 0), label_prefix="EF ")
#     early_panel = make_panel_title(early_panel, "Early Fusion")

#     row = hstack_resize([ir_panel, early_panel], target_h=TARGET_H)

#     footer = np.zeros((48, row.shape[1], 3), dtype=np.uint8)
#     cv2.putText(
#         footer, footer_text, (10, 32),
#         cv2.FONT_HERSHEY_SIMPLEX, 0.75,
#         (255, 255, 255), 2, cv2.LINE_AA
#     )

#     return cv2.vconcat([row, footer])


# # ============================================================
# # MAIN
# # ============================================================

# def main():
#     ensure_dir(OUT_DIR)
#     ensure_dir(os.path.join(OUT_DIR, "figures"))

#     rgb_map = stem_to_path_dict(RGB_DIR)
#     ir_map = stem_to_path_dict(IR_DIR)
#     common_stems = sorted(set(rgb_map.keys()) & set(ir_map.keys()))

#     ir_preds_map = load_predictions_json(IR_JSON_PATH, CONF_THRES)
#     early_preds_map = load_predictions_json(EARLY_JSON_PATH, CONF_THRES)

#     rescued_rows = []
#     image_rows = []

#     # global metrics accumulators
#     total_ir_tp = total_ir_fp = total_ir_fn = 0
#     total_ef_tp = total_ef_fp = total_ef_fn = 0

#     for stem in common_stems:
#         rgb_path = rgb_map[stem]
#         ir_path = ir_map[stem]

#         rgb = cv2.imread(rgb_path)
#         ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
#         if rgb is None or ir is None:
#             continue

#         img_h, img_w = rgb.shape[:2]

#         gt_boxes = parse_yolo_gt(
#             os.path.join(GT_LABELS_DIR, f"{stem}.txt"),
#             img_w, img_h
#         )

#         ir_preds = ir_preds_map.get(stem, [])
#         early_preds = early_preds_map.get(stem, [])

#         ir_match, ir_unmatched_gt, ir_unmatched_pred = greedy_match(gt_boxes, ir_preds, IOU_THRES)
#         ef_match, ef_unmatched_gt, ef_unmatched_pred = greedy_match(gt_boxes, early_preds, IOU_THRES)

#         ir_tp = len(ir_match)
#         ir_fp = len(ir_unmatched_pred)
#         ir_fn = len(ir_unmatched_gt)

#         ef_tp = len(ef_match)
#         ef_fp = len(ef_unmatched_pred)
#         ef_fn = len(ef_unmatched_gt)

#         total_ir_tp += ir_tp
#         total_ir_fp += ir_fp
#         total_ir_fn += ir_fn

#         total_ef_tp += ef_tp
#         total_ef_fp += ef_fp
#         total_ef_fn += ef_fn

#         tp_gain = ef_tp - ir_tp
#         fn_reduction = ir_fn - ef_fn
#         fp_delta = ef_fp - ir_fp

#         image_gain_score = 2.0 * tp_gain + fn_reduction - 0.5 * max(0, fp_delta)

#         image_rows.append({
#             "stem": stem,
#             "ir_tp": ir_tp,
#             "ir_fp": ir_fp,
#             "ir_fn": ir_fn,
#             "early_tp": ef_tp,
#             "early_fp": ef_fp,
#             "early_fn": ef_fn,
#             "tp_gain": tp_gain,
#             "fn_reduction": fn_reduction,
#             "fp_delta": fp_delta,
#             "image_gain_score": image_gain_score
#         })

#         rescued_gt_indices = [gi for gi in ef_match.keys() if gi not in ir_match]

#         for gi in rescued_gt_indices:
#             gt = gt_boxes[gi]
#             area_ratio = gt.area / float(img_w * img_h)
#             border_flag = (
#                 gt.cx / img_w < 0.15 or gt.cx / img_w > 0.85 or
#                 gt.cy / img_h < 0.15 or gt.cy / img_h > 0.85
#             )

#             stats = crop_stats_ir(ir, gt)
#             nn_dist = nearest_neighbor_distance(gi, gt_boxes, img_w, img_h)
#             neighbor_count = count_neighbors(gi, gt_boxes, img_w, img_h, radius=0.10)

#             pattern = assign_pattern(
#                 area_ratio=area_ratio,
#                 border_flag=border_flag,
#                 ir_std=stats["ir_std"],
#                 neighbor_count=neighbor_count,
#                 nn_dist=nn_dist
#             )

#             rescued_rows.append({
#                 "stem": stem,
#                 "gt_idx": gi,
#                 "pattern": pattern,
#                 "area_ratio": area_ratio,
#                 "border_flag": int(border_flag),
#                 "ir_mean": stats["ir_mean"],
#                 "ir_std": stats["ir_std"],
#                 "neighbor_count": neighbor_count,
#                 "nearest_neighbor_dist": nn_dist,
#                 "image_gain_score": image_gain_score
#             })

#     rescued_df = pd.DataFrame(rescued_rows)
#     image_df = pd.DataFrame(image_rows)

#     rescued_df.to_csv(os.path.join(OUT_DIR, "rescued_instances.csv"), index=False)
#     image_df.to_csv(os.path.join(OUT_DIR, "image_level_comparison.csv"), index=False)

#     # -----------------------------
#     # Metrics summary for the table
#     # -----------------------------
#     def prec(tp, fp):
#         return tp / (tp + fp) if (tp + fp) > 0 else 0.0

#     def rec(tp, fn):
#         return tp / (tp + fn) if (tp + fn) > 0 else 0.0

#     metrics_df = pd.DataFrame([
#         {
#             "Method": "IR-only",
#             "Precision": prec(total_ir_tp, total_ir_fp),
#             "Recall": rec(total_ir_tp, total_ir_fn),
#             "TP": total_ir_tp,
#             "FP": total_ir_fp,
#             "FN": total_ir_fn
#         },
#         {
#             "Method": "Early Fusion",
#             "Precision": prec(total_ef_tp, total_ef_fp),
#             "Recall": rec(total_ef_tp, total_ef_fn),
#             "TP": total_ef_tp,
#             "FP": total_ef_fp,
#             "FN": total_ef_fn
#         }
#     ])
#     metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)

#     # -----------------------------
#     # Pattern summary
#     # -----------------------------
#     if not rescued_df.empty:
#         summary = (
#             rescued_df.groupby("pattern")
#             .agg(
#                 rescued_cases=("stem", "count"),
#                 unique_images=("stem", "nunique"),
#                 mean_area_ratio=("area_ratio", "mean"),
#                 mean_ir_std=("ir_std", "mean"),
#                 mean_neighbor_count=("neighbor_count", "mean"),
#                 mean_gain_score=("image_gain_score", "mean")
#             )
#             .sort_values("rescued_cases", ascending=False)
#             .reset_index()
#         )
#     else:
#         summary = pd.DataFrame(columns=[
#             "pattern", "rescued_cases", "unique_images",
#             "mean_area_ratio", "mean_ir_std", "mean_neighbor_count", "mean_gain_score"
#         ])

#     summary.to_csv(os.path.join(OUT_DIR, "patterns_summary.csv"), index=False)

#     print("\n=== Metrics summary ===")
#     print(metrics_df)
#     print("\n=== Pattern summary ===")
#     print(summary)

#     # -----------------------------
#     # Save top examples per pattern
#     # -----------------------------
#     if not rescued_df.empty:
#         for pattern in summary["pattern"].tolist():
#             pattern_dir = os.path.join(OUT_DIR, "figures", pattern)
#             ensure_dir(pattern_dir)

#             pattern_rescues = rescued_df[rescued_df["pattern"] == pattern]
#             candidate_stems = (
#                 pattern_rescues.groupby("stem")["image_gain_score"]
#                 .max()
#                 .sort_values(ascending=False)
#                 .head(TOP_K_PER_PATTERN)
#                 .index.tolist()
#             )

#             for stem in candidate_stems:
#                 rgb = cv2.imread(rgb_map[stem])
#                 ir = cv2.imread(ir_map[stem], cv2.IMREAD_GRAYSCALE)
#                 img_h, img_w = rgb.shape[:2]

#                 gt_boxes = parse_yolo_gt(
#                     os.path.join(GT_LABELS_DIR, f"{stem}.txt"),
#                     img_w, img_h
#                 )
#                 ir_preds = ir_preds_map.get(stem, [])
#                 early_preds = early_preds_map.get(stem, [])

#                 ir_match, _, _ = greedy_match(gt_boxes, ir_preds, IOU_THRES)
#                 ef_match, _, _ = greedy_match(gt_boxes, early_preds, IOU_THRES)
#                 rescued_gt_indices = [gi for gi in ef_match.keys() if gi not in ir_match]

#                 footer_text = f"{stem} | pattern: {pattern} | rescued GT: {len(rescued_gt_indices)}"
#                 canvas = make_two_panel_figure(
#                     rgb=rgb,
#                     ir=ir,
#                     gt_boxes=gt_boxes,
#                     ir_preds=ir_preds,
#                     early_preds=early_preds,
#                     rescued_gt_indices=rescued_gt_indices,
#                     footer_text=footer_text
#                 )
#                 cv2.imwrite(os.path.join(pattern_dir, f"{stem}.jpg"), canvas)

#     # -----------------------------
#     # Save top overall examples
#     # -----------------------------
#     top_dir = os.path.join(OUT_DIR, "figures", "top_overall")
#     ensure_dir(top_dir)

#     top_overall = image_df.sort_values("image_gain_score", ascending=False).head(TOP_K_OVERALL)["stem"].tolist()

#     for stem in top_overall:
#         rgb = cv2.imread(rgb_map[stem])
#         ir = cv2.imread(ir_map[stem], cv2.IMREAD_GRAYSCALE)
#         img_h, img_w = rgb.shape[:2]

#         gt_boxes = parse_yolo_gt(
#             os.path.join(GT_LABELS_DIR, f"{stem}.txt"),
#             img_w, img_h
#         )
#         ir_preds = ir_preds_map.get(stem, [])
#         early_preds = early_preds_map.get(stem, [])

#         ir_match, _, _ = greedy_match(gt_boxes, ir_preds, IOU_THRES)
#         ef_match, _, _ = greedy_match(gt_boxes, early_preds, IOU_THRES)
#         rescued_gt_indices = [gi for gi in ef_match.keys() if gi not in ir_match]

#         row = image_df[image_df["stem"] == stem].iloc[0]
#         footer_text = (
#             f"{stem} | TP gain: {row['tp_gain']} | "
#             f"FN reduction: {row['fn_reduction']} | FP delta: {row['fp_delta']}"
#         )

#         canvas = make_two_panel_figure(
#             rgb=rgb,
#             ir=ir,
#             gt_boxes=gt_boxes,
#             ir_preds=ir_preds,
#             early_preds=early_preds,
#             rescued_gt_indices=rescued_gt_indices,
#             footer_text=footer_text
#         )
#         cv2.imwrite(os.path.join(top_dir, f"{stem}.jpg"), canvas)

#     # -----------------------------
#     # Interpretation text
#     # -----------------------------
#     suggestions = {
#         "crowded_scene": "Early fusion helps separate close pedestrians when IR blobs overlap or merge.",
#         "small_far_targets": "Early fusion recovers small or distant pedestrians by adding visible structure cues.",
#         "low_ir_contrast": "Early fusion helps when thermal contrast is weak or faint in IR.",
#         "border_hard_case": "Early fusion improves detections near borders where single-modality evidence is incomplete.",
#         "general_complementarity": "Early fusion benefits from RGB context while preserving thermal saliency."
#     }

#     with open(os.path.join(OUT_DIR, "pattern_interpretation.txt"), "w") as f:
#         for k, v in suggestions.items():
#             f.write(f"{k}: {v}\n")

#     print(f"\nDone. Outputs saved to: {OUT_DIR}")


# if __name__ == "__main__":
#     main()