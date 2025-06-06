from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = ROOT_DIR / "config.ini"

SHOGI_NN_DIR = ROOT_DIR / "ShogiNeuralNetwork"
DATASETS_DIR = ROOT_DIR / "datasets"

FIGURE_YOLO_DATASET_PATH = DATASETS_DIR / "figure_classification"
DIRECTION_YOLO_DATASET_PATH = DATASETS_DIR / "direction_classification"
BOARD_YOLO_DATASET_PATH = DATASETS_DIR / "board_segmentation" / "data.yaml"
CELLS_DATASET_PATH = DATASETS_DIR / "figure_direction_classification"

MODELS_DIR = ROOT_DIR / "models"
FIGURE_CLASSIFICATION_YOLO_MODEL_PATH = MODELS_DIR / "figure_classifier.pt"
DIRECTION_CLASSIFICATION_YOLO_MODEL_PATH = MODELS_DIR / "direction_classifier.pt"
BOARD_SEGMENTATION_YOLO_MODEL_PATH = MODELS_DIR / "board_segmenter.pt"
BOARD_SEGMENTATION_ONNX_MODEL_PATH = MODELS_DIR / "board_segmenter.onnx"
MIXED_MODEL_KERAS_PATH = MODELS_DIR / "mixed.keras"
MIXED_MODEL_EXPORT_PATH = MODELS_DIR / "mixed"
MIXED_MODEL_ONNX_PATH = MODELS_DIR / "mixed.onnx"
MIXED_MODEL_TFLITE_PATH = MODELS_DIR / "mixed.tflite"

IMGS_DIR = ROOT_DIR / "img"
FIGURE_ICONS_DIR = IMGS_DIR / "figures icons"
ICONS_DIR = IMGS_DIR / "Icons"

SOUNDS_DIR_PATH = ROOT_DIR / "sounds"
ALARM_PATH = SOUNDS_DIR_PATH / "alarm.wav"
