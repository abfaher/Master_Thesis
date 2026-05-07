from ultralytics import YOLO


def main():
    model = YOLO("yolov8l.yaml")  # baseline

    model.train(
        data="/workspace/PixelFusion/M1/fused_alpha_04.yaml",
        epochs=100,
        patience=20, # Early stopping
        imgsz=640,
        batch=8,
        workers=4,
        amp=False,
        project="/workspace/PixelFusion/M1/experiments",
        name="yolov8l_fused_alpha_04",
        pretrained=False,
    )


if __name__ == "__main__":
    main()