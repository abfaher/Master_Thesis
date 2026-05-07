## Here's the commands order to execute the program:

### Method 1: Union + NMS

cd /workspace/LateFusion
python Utils/predictions_to_coco.py
python M1_Union/union_nms.py
python Utils/eval_fusion.py -> to evaluate Union + NMS


### Method 2: WBF

cd /workspace/LateFusion
python Utils/predictions_to_coco.py
python M2_WBF/wbf.py
python Utils/eval_fusion.py --fused M2_WBF/predictions_fused_wbf.json -> to evaluate WBF