import shutil

from ultralytics import YOLO

from config import paths

model = YOLO("yolo11n-seg.pt")

if __name__ == '__main__':
    model.train(
        data=paths.BOARD_YOLO_DATASET_PATH,
        epochs=100,
        imgsz=640,
    )
    trained_model_path = model.export(format="onnx")
    shutil.move(trained_model_path, paths.BOARD_SEGMENTATION_YOLO_MODEL_PATH)
