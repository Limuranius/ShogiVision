from Elements import *
from Elements.CornerDetectors.YOLOSegmentationCornerDetector import YOLOSegmentationCornerDetector
from Elements.ImageGetters import Photo
from config import paths, GLOBAL_CONFIG


def default_recognizer() -> Recognizers.Recognizer:
    return RecognizerONNX(paths.MIXED_MODEL_ONNX_PATH)
    # return RecognizerYOLO(
    #     figure_model_path=paths.FIGURE_CLASSIFICATION_YOLO_MODEL_PATH,
    #     direction_model_path=paths.DIRECTION_CLASSIFICATION_YOLO_MODEL_PATH,
    # )


def default_corner_detector():
    return YOLOSegmentationCornerDetector()
    # return YOLOONNX()


def default_board_splitter():
    return BoardSplitter(
        image_getter=Photo(),
        corner_getter=default_corner_detector(),
    )


def hsv_corner_detector():
    hsv_low = (
        GLOBAL_CONFIG.HSVThreshold.h_low,
        GLOBAL_CONFIG.HSVThreshold.s_low,
        GLOBAL_CONFIG.HSVThreshold.v_low,
    )
    hsv_high = (
        GLOBAL_CONFIG.HSVThreshold.h_high,
        GLOBAL_CONFIG.HSVThreshold.s_high,
        GLOBAL_CONFIG.HSVThreshold.v_high,
    )
    return CornerDetectors.HSVThresholdCornerDetector(hsv_low, hsv_high)


def empty_reader():
    return ShogiBoardReader(
        board_splitter=BoardSplitter(
            image_getter=None,
            corner_getter=None,
            inventory_detector=None
        ),
        recognizer=default_recognizer(),
        memorizer=None
    )


def image_reader():
    reader = ShogiBoardReader(
        board_splitter=BoardSplitter(
            image_getter=Photo(),
            corner_getter=default_corner_detector(),
        ),
        recognizer=default_recognizer()
    )
    return reader


def book_reader():
    reader = ShogiBoardReader(
        board_splitter=BoardSplitter(
            image_getter=ImageGetters.Photo(),
            corner_getter=BookCornerDetector(),
            inventory_detector=BookInventoryDetector()
        ),
        recognizer=default_recognizer()
    )
    return reader


def camera_reader():
    reader = ShogiBoardReader(
        board_splitter=BoardSplitter(
            image_getter=ImageGetters.Camera(),
            corner_getter=default_corner_detector(),
            inventory_detector=None
        ),
        recognizer=default_recognizer()
    )
    return reader
