"""
Runs the full baseline pipeline in order:
0) Build IR train/val/test filelists
1) Convert VOC XML -> YOLO txt labels (IR/train|test/labels)
2) Check that every image has a matching label
3) Train YOLOv8-L on LLVIP-IR
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS   = REPO_ROOT / "scripts"

def run(step_name, cmd):
    print(f"\n=== [{step_name}] ===")
    print(" ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"[{step_name}] failed with return code {r.returncode}")

def main():
    py = sys.executable  # use the Python version of the active env

    # 0) Make IR lists (safe to re-run -> it overwrites the 3 txt files)
    run("MAKE IR LISTS",
        [py, str(SCRIPTS / "make_lists_ir.py")]
    )

    # 1) Convert labels
    run("CONVERT LABELS",
        [py, str(SCRIPTS / "to_yolo_labels_format.py")]
    )

    # 2) Check labels
    run("CHECK LABELS",
        [py, str(SCRIPTS / "check_labels.py")]
    )

    # 3) Train
    run("TRAIN YOLOv8-L",
        [py, str(SCRIPTS / "train_ir.py")]
    )

if __name__ == "__main__":
    main()
