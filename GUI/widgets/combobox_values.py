import Elements
from extra import factories
from extra import utils


def corner_detector():
    return "Corner detector", [
        (
            "Neural Network",
            "Searches for board position using neural network\nUse it if you're unsure what to choose",
            # Elements.YOLOSegmentationCornerDetector(),
            Elements.YOLOONNX(),
        ),
        (
            "Edge detection",
            "Detects board corners using edge detection",
            Elements.CoolCornerDetector()
        ),
        (
            "Book",
            "Works really good in e-books",
            Elements.BookCornerDetector()
        ),
        (
            "HSV Markers",
            "Detects colored markers using HSV thresholding",
            factories.hsv_corner_detector()
        ),
        (
            "Manual",
            "Set corners manually. If nothing else is working...",
            Elements.HardcodedCornerDetector()
        ),
    ]


def inventory_detector():
    return "Inventory detector", [
        (
            "None",
            "Don't use inventory detector",
            None
        ),
        (
            "Book",
            "Detects inventories of e-books' boards",
            Elements.InventoryDetectors.BookInventoryDetector()
        ),
    ]


def memorizer():
    return "Memorizer", [
        (
            "None",
            "Don't use memorizer",
            None
        ),
        (
            "Memorizer",
            "Use memorizer",
            Elements.BoardMemorizer()
        ),
    ]


def image_getter():
    return "Image source", [
        (
            "Photo",
            "Photo of board",
            Elements.ImageGetters.Photo()
        ),
        (
            "Video",
            "Recorded video of board",
            Elements.ImageGetters.VideoFile()
        ),
        (
            "Camera",
            "Use camera connected to computer",
            Elements.ImageGetters.Camera()
        ),
    ]


# Precalculating cameras on first start as it takes some time
cams = []
for cam_id in utils.get_available_cam_ids():
    cams.append((
        str(cam_id),
        f"Camera {cam_id}",
        Elements.ImageGetters.Camera(cam_id)
    ))


def cameras():
    return "Camera", cams
