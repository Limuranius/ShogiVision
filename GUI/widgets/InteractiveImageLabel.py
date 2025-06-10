import math
from collections.abc import Callable

import cv2
import numpy as np
from PyQt5.QtGui import QMouseEvent

from GUI.widgets.ImageLabel import ImageLabel
from extra.types import Box, ImageNP


class InteractiveImageLabel(ImageLabel):
    """
    Image label with interactive areas that can be clicked on and call some callback function
    """

    # Area on image (full size), callback function that is called when clicked on
    _clickable_boxes: list[tuple[Box, Callable[[], None]]]
    _pixel_indices: np.ndarray  # For each pixel tells index of bounding box
    _selected_box_i: int  # index of currently selected box. 0 if no boxes selected

    _original_image: ImageNP

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self.set_clickable_boxes([])
        self._selected_box_i = 0

    def set_image(self, image: ImageNP) -> None:
        super().set_image(image)
        self._original_image = image

    def set_clickable_boxes(
            self,
            boxes: list[tuple[Box, Callable[[], None]]],
            outline_boxes=False,
    ) -> None:
        self._clickable_boxes = boxes
        self._pixel_indices = np.zeros(shape=self._image.shape[:2], dtype=int)
        if outline_boxes:
            for (x, y, w, h), _ in boxes:
                cv2.rectangle(self._original_image, (x, y), (x + w, y + h), (0, 0, 255),
                              thickness=math.ceil(sum(self._original_image.shape[:2]) / 1500))
        for i, ((x, y, w, h), _) in enumerate(boxes):
            self._pixel_indices[y: y + h, x: x + w] = i + 1
        super().set_image(self._original_image)

    def add_clickable_box(self, box: Box, callback: Callable[[], None]) -> None:
        boxes = self._clickable_boxes.copy()
        boxes.append((box, callback))
        self.set_clickable_boxes(boxes)

    def remove_clickable_box(self, box_i: int) -> None:
        boxes = self._clickable_boxes.copy()
        del boxes[box_i]
        self.set_clickable_boxes(boxes)

    def mouseMoveEvent(self, ev: QMouseEvent):
        """When mouse hovers over widget"""
        i = self.__mouse_pos_to_box_index(ev.x(), ev.y())
        if i != self._selected_box_i:
            self._selected_box_i = i
            if i == 0:
                super().set_image(self._original_image)
                return

            frame = self._original_image.copy()
            x, y, w, h = self._clickable_boxes[i - 1][0]
            super().set_image(self.add_selection(frame, x, y, w, h))

    def add_selection(self, image: ImageNP, x: int, y: int, w: int, h: int) -> ImageNP:
        # Making selected area darker
        image[y: y + h, x: x + w] //= 5
        image[y: y + h, x: x + w] *= 4
        return image

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        super().mousePressEvent(ev)
        i = self.__mouse_pos_to_box_index(ev.x(), ev.y())
        if i != 0:
            self._clickable_boxes[i - 1][1]()

    def __mouse_pos_to_box_index(self, x: int, y: int) -> int:
        h, w = self._image.shape[:2]
        _, (orig_x, orig_y) = self._label_coord_to_image_coord(x, y)
        if 0 <= orig_x < w and 0 <= orig_y < h:
            return int(self._pixel_indices[orig_y, orig_x])
        return 0  # mouse outside of image
