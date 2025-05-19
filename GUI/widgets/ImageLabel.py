from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
from extra.types import ImageNP
from extra.utils import generate_random_image


class ImageLabel(QLabel):
    """
    Widget that shows image
    """

    # Emits on mouse click
    # Emits (x, y) on scaled image and (x, y) on original image
    clicked = pyqtSignal(int, int, int, int)

    _image: ImageNP
    _pixmap: QPixmap
    _scaled_pixmap: QPixmap

    def __init__(self, *args, min_size=(1, 1), **kwargs):
        super().__init__(*args, **kwargs)
        self.set_image(generate_random_image(500, 500, 3))
        self.setMinimumSize(*min_size)
        self.setStyleSheet("background-color:black;")
        self.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

    def set_image(self, image: ImageNP) -> None:
        self._image = image
        self.__update_pixmap_from_numpy()
        self.__rescale_pixmap()

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        """Actions when mouse clicks on image. Emits :clicked signal"""
        (view_x, view_y), (orig_x, orig_y) = self._label_coord_to_image_coord(ev.x(), ev.y())
        self.clicked.emit(view_x, view_y, orig_x, orig_y)

    def _label_coord_to_image_coord(self, x: int, y: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Converts coordinate local to label to coordinate local to image
        Needed when image keeps aspect ratio and paddings occur on sides
        Returns (x, y) on scaled image and (x, y) on original image
        """
        label_h, label_w = self.height(), self.width()
        pixmap_h, pixmap_w = self._scaled_pixmap.height(), self._scaled_pixmap.width()
        padding_x = (label_w - pixmap_w) // 2
        padding_y = (label_h - pixmap_h) // 2

        view_x, view_y = x - padding_x, y - padding_y
        orig_h, orig_w = self._pixmap.height(), self._pixmap.width()
        h_factor = orig_h / pixmap_h
        w_factor = orig_w / pixmap_w
        orig_x = int(view_x * w_factor)
        orig_y = int(view_y * h_factor)
        return (view_x, view_y), (orig_x, orig_y)


    def resizeEvent(self, a0) -> None:
        """When widget is resized"""
        self.__rescale_pixmap()

    def __rescale_pixmap(self) -> None:
        """Updates image and resizes it to widget size"""
        self._scaled_pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self._scaled_pixmap)

    def __update_pixmap_from_numpy(self) -> None:
        height, width = self._image.shape[:2]
        qimg = QImage(self._image.data, width, height, width * 3, QImage.Format_BGR888)
        self._pixmap = QPixmap(qimg)
