import sys
sys.path.insert(0, "/workspace/EarlyFusion/ultralytics")

from ultralytics import YOLO


def main():
    model = YOLO("/workspace/EarlyFusion/yolov8l_4ch.yaml")
    model.train(
    data="/workspace/EarlyFusion/llvip.yaml",
    epochs=100,
    patience=20, # Early stopping
    imgsz=640,
    batch=8,  # it was working with 2 (in case 8 doesn't work for the second tentaive)
    workers=4,
    amp=False,
    project="/workspace/EarlyFusion/experiments",
    name="yolov8l_earlyfusion",
    pretrained=False,
)

if __name__ == "__main__":
    main()