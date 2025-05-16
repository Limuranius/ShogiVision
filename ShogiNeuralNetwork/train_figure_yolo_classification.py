import os.path

from ultralytics import YOLO
from config import paths
import shutil

model = YOLO("yolo11s-cls.pt")

if __name__ == '__main__':
    result = model.train(
        data=paths.FIGURE_YOLO_DATASET_PATH,
        epochs=200,
        imgsz=64,
        batch=64,

        shear=10,
        degrees=35,
        bgr=0.3,
        flipud=0.5,
    )
    print(dir(model))
    print(dir(result))
    # trained_model_path = model.export(format="onnx")
    # shutil.move(trained_model_path, paths.FIGURE_CLASSIFICATION_MODEL_PATH)

    trained_model_path = os.path.join(result.save_dir, "weights", "best.pt")
    shutil.copy(trained_model_path, paths.FIGURE_CLASSIFICATION_YOLO_MODEL_PATH)