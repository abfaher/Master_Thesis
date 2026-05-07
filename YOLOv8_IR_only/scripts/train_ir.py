from ultralytics import YOLO
from pathlib import Path

def main():
    model = YOLO("yolov8l.pt")  # stock YOLOv8-L, expects 3-channel input

    model.train(
        data="configs/llvip_ir.yaml",
        epochs=100,
        imgsz=640,
        batch=32,
    )

if __name__ == "__main__":
    main()