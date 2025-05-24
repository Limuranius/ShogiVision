import os.path

import cv2
import numpy as np
from abc import ABC, abstractmethod
from extra.utils import generate_random_image
from extra.types import ImageNP


ROTATIONS = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]


class ImageGetter(ABC):
    # Rotation that is applied to image
    # One of: None (no rotation), cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180
    rotation: int | None = None

    @abstractmethod
    def get_image(self) -> ImageNP:
        pass

    def set_rotation(self, rotation_code: int) -> None:
        self.rotation = rotation_code

    def rotate(self):
        next_rotation = ROTATIONS[(ROTATIONS.index(self.rotation) + 1) % 4]
        self.set_rotation(next_rotation)

    def rotate_image(self, img: ImageNP):
        if self.rotation is None:
            return img
        else:
            return cv2.rotate(img, self.rotation)


class Photo(ImageGetter):
    img: np.ndarray

    def __init__(self, img: str | ImageNP = ""):
        if isinstance(img, str):
            if os.path.exists(img):
                self.img = cv2.imread(img)
            else:
                self.img = None
        else:
            self.img = img
        if self.img is None:
            self.img = generate_random_image(500, 500, 3)

    def get_image(self) -> ImageNP:
        return self.rotate_image(self.img.copy())

    def __copy__(self):
        new_img = Photo(self.img.copy())
        return new_img


class Camera(ImageGetter):
    video: cv2.VideoCapture

    def __init__(
            self,
            cam_id: int = 0,
            width_height: tuple[int, int] = None,
    ):
        self.video = cv2.VideoCapture(cam_id)

        if width_height is not None:
            width, height = width_height
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_image(self) -> ImageNP:
        ret, frame = self.video.read()
        if ret:
            return self.rotate_image(frame)
        else:
            return generate_random_image(500, 500, 3)


class Video(ImageGetter):
    video: cv2.VideoCapture
    finished_playing: bool
    __path: str

    def __init__(self, video_path: str = ""):
        self.video = cv2.VideoCapture(video_path)
        self.finished_playing = False
        self.__path = video_path

    def get_image(self) -> ImageNP:
        ret, frame = self.video.read()
        if ret:
            return self.rotate_image(frame)
        else:
            return generate_random_image(500, 500, 3)

    def restart(self) -> None:
        self.video = cv2.VideoCapture(self.__path)

    def __copy__(self):
        return Video(self.__path)

    def frames_count(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
