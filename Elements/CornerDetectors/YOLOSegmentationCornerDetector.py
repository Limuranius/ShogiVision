import cv2
from ultralytics import YOLO

from extra.utils import order_points
from .CornerDetector import CornerDetector
from config import paths
from extra.types import ImageNP, Corners

model = YOLO(paths.BOARD_SEGMENTATION_YOLO_MODEL_PATH, task="segment")


class YOLOSegmentationCornerDetector(CornerDetector):
    def get_corners(self, image: ImageNP) -> Corners:
        results = model(image, verbose=False)
        for result in results:
            if result.masks is None:
                h, w = image.shape[:2]
                return (0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)
            poly = result.masks.xy[0].astype(int)
            arclen = cv2.arcLength(poly, True)
            poly = cv2.approxPolyDP(poly, 0.02 * arclen, True)
            return order_points(poly[:, 0])
