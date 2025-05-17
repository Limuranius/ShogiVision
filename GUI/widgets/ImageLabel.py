from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
from extra.types import ImageNP
from extra.utils import generate_random_image


class ImageLabel(QLabel):
    # Emits on mouse click
    # Emits (x, y) on scaled image and (x, y) on original image
    clicked = pyqtSignal(int, int, int, int)

    __pixmap: QPixmap
    __scaled_pixmap: QPixmap

    def __init__(self, *args, min_size=(1, 1), **kwargs):
        super().__init__(*args, **kwargs)
        self.set_image(generate_random_image(500, 500, 3))
        self.setMinimumSize(*min_size)
        self.setStyleSheet("background-color:black;")
        self.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

    def set_image(self, image: ImageNP) -> None:
        height, width = image.shape[:2]
        qimg = QImage(image.data, width, height, width * 3, QImage.Format_BGR888)
        self.__pixmap = QPixmap(qimg)
        self.__update_pixmap()

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        label_h, label_w = self.height(), self.width()
        pixmap_h, pixmap_w = self.__scaled_pixmap.height(), self.__scaled_pixmap.width()
        padding_x = (label_w - pixmap_w) // 2
        padding_y = (label_h - pixmap_h) // 2

        view_x, view_y = ev.x() - padding_x, ev.y() - padding_y
        orig_h, orig_w = self.__pixmap.height(), self.__pixmap.width()
        h_factor = orig_h / pixmap_h
        w_factor = orig_w / pixmap_w
        orig_x = int(view_x * w_factor)
        orig_y = int(view_y * h_factor)
        self.clicked.emit(view_x, view_y, orig_x, orig_y)

    def resizeEvent(self, a0):
        self.__update_pixmap()

    def __update_pixmap(self):
        self.__scaled_pixmap = self.__pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self.__scaled_pixmap)
