from ultralytics import YOLO

def main():
    model = YOLO("yolov8l.yaml")  # baseline

    model.train(
        data="llvip_rgb.yaml",
        epochs=100,
        patience=20, # Early stopping
        imgsz=640,
        batch=8,
        workers=4,
        amp=False,
        project="/workspace/YOLOv8_RGB_only/experiments",
        name="yolov8l_rgb_only",
        pretrained=False,
    )

if __name__ == "__main__":
    main()