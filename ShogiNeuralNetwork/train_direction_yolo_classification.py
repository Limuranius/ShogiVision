from ultralytics import YOLO
from config import paths
import shutil

model = YOLO("yolo11n-cls.pt")

if __name__ == '__main__':
    model.train(
        data=paths.DIRECTION_YOLO_DATASET_PATH,
        epochs=20,
        imgsz=64
    )
    trained_model_path = model.export(format="onnx")
    shutil.move(trained_model_path, paths.DIRECTION_CLASSIFICATION_YOLO_MODEL_PATH)