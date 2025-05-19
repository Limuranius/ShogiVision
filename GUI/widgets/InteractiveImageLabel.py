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
    _frames: list[ImageNP]
    _curr_frame_i: int
    _pixel_indices: np.ndarray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self.set_clickable_boxes([])
        self._curr_frame_i = 0

    def set_clickable_boxes(
            self,
            boxes: list[tuple[Box, Callable[[], None]]],
            outline_boxes=True,
    ) -> None:
        self._clickable_boxes = boxes
        self._pixel_indices = np.zeros(shape=self._image.shape[:2], dtype=int)
        base_frame = self._image.copy()
        self._frames = [base_frame]
        if outline_boxes:
            for (x, y, w, h), _ in boxes:
                cv2.rectangle(base_frame, (x, y), (x + w, y + h), (0, 0, 255),
                              thickness=math.ceil(sum(base_frame.shape[:2]) / 1500))
        for i, ((x, y, w, h), _) in enumerate(boxes):
            self._pixel_indices[y: y + h, x: x + w] = i + 1
            frame = base_frame.copy()
            # Making selected area darker
            frame[y: y + h, x: x + w] //= 5
            frame[y: y + h, x: x + w] *= 4
            self._frames.append(frame)

        self.set_image(base_frame)

    def mouseMoveEvent(self, ev: QMouseEvent):
        """When mouse hovers over widget"""
        i = self.__mouse_pos_to_frame_index(ev.x(), ev.y())
        if i != self._curr_frame_i:
            self._curr_frame_i = i
            self.set_image(self._frames[i])

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        super().mousePressEvent(ev)
        i = self.__mouse_pos_to_frame_index(ev.x(), ev.y())
        if i != 0:
            self._clickable_boxes[i - 1][1]()

    def __mouse_pos_to_frame_index(self, x: int, y: int) -> int:
        h, w = self._image.shape[:2]
        _, (orig_x, orig_y) = self._label_coord_to_image_coord(x, y)
        if 0 <= orig_x < w and 0 <= orig_y < h:
            return int(self._pixel_indices[orig_y, orig_x])
        return 0  # mouse outside of image
