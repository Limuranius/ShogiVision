from config import GLOBAL_CONFIG
from .ImageGetters import ImageGetter
from Elements.CornerDetectors.CornerDetector import CornerDetector
from Elements.InventoryDetectors import InventoryDetector
from extra import utils
from extra.types import ImageNP
import numpy as np
import cv2
import copy


class BoardSplitter:
    __image_getter: ImageGetter
    __corner_detector: CornerDetector
    __inventory_detector: InventoryDetector

    def __init__(self,
                 image_getter: ImageGetter,
                 corner_getter: CornerDetector,
                 inventory_detector: InventoryDetector = None):
        self.__image_getter = image_getter
        self.__corner_detector = corner_getter
        self.__inventory_detector = inventory_detector

    def get_board_image_no_perspective(self,
                                       draw_grid: bool = False) -> ImageNP:
        """Returns image of board without perspective and surroundings"""
        full_img = self.__image_getter.get_image()
        corners = self.__corner_detector.get_corners(full_img)
        img_no_persp = utils.remove_perspective(full_img, np.array(corners))
        if draw_grid:
            h, w = img_no_persp.shape[:2]
            grid_thickness = int((h + w) / 2 * GLOBAL_CONFIG.Visuals.lines_thickness_fraction) + 1
            for x in np.linspace(0, w, num=10, dtype=np.int_):
                cv2.line(img_no_persp, [x, 0], [x, h], color=[0, 255, 0], thickness=grid_thickness)
            for y in np.linspace(0, h, num=10, dtype=np.int_):
                cv2.line(img_no_persp, [0, y], [w, y], color=[0, 255, 0], thickness=grid_thickness)
        return img_no_persp

    def get_board_cells(self) -> list[list[ImageNP]]:
        """Returns 2D 9x9 list with images of cells"""
        board_img = self.get_board_image_no_perspective()
        return self.__get_board_cells(board_img)

    def __get_board_cells(self, board_img: ImageNP) -> list[list[ImageNP]]:
        """Splits image into 81 (9x9) images of each cell"""

        height = board_img.shape[0]
        width = board_img.shape[1]
        x_step = width // 9
        y_step = height // 9

        result = [[None for _ in range(9)] for __ in range(9)]
        for y in range(1, 10):
            for x in range(1, 10):
                x_start = x_step * (x - 1)
                x_end = x_step * x
                y_start = y_step * (y - 1)
                y_end = y_step * y
                cell_img = board_img[y_start: y_end, x_start: x_end]
                result[y - 1][x - 1] = cell_img
        return result

    def get_full_img(
            self,
            show_borders: bool = False,
            show_grid: bool = False,
            show_inventories: bool = False
    ) -> ImageNP:
        full_img = self.__image_getter.get_image().copy()
        color = [0, 255, 0]
        h, w = full_img.shape[:2]
        thickness = int((h + w) / 2 * GLOBAL_CONFIG.Visuals.lines_thickness_fraction) + 1
        corners = np.array(self.__corner_detector.get_corners(full_img))
        if show_borders:
            cv2.polylines(full_img, [corners], True, color, thickness=thickness)
        if show_grid:
            top_points = np.linspace(corners[0], corners[1], num=10, dtype=np.int_)
            right_points = np.linspace(corners[1], corners[2], num=10, dtype=np.int_)
            bottom_points = np.linspace(corners[2], corners[3], num=10, dtype=np.int_)
            left_points = np.linspace(corners[3], corners[0], num=10, dtype=np.int_)

            for p1, p2 in zip(top_points, reversed(bottom_points)):
                cv2.line(full_img, p1, p2, color, thickness)
            for p1, p2 in zip(left_points, reversed(right_points)):
                cv2.line(full_img, p1, p2, color, thickness)
        if show_inventories and self.__inventory_detector is not None:
            i1_corners, i2_corners = self.__inventory_detector.get_inventories_corners(full_img)
            i1_corners = np.array(i1_corners)
            i2_corners = np.array(i2_corners)
            cv2.polylines(full_img, [i1_corners, i2_corners], True, color, thickness=thickness)
        return full_img

    def get_inventory_cells(self) -> tuple[list[ImageNP], list[ImageNP]]:
        img = self.get_full_img()
        i1_imgs, i2_imgs = self.__inventory_detector.get_figure_images(img)
        return i1_imgs, i2_imgs

    def __copy__(self):
        return BoardSplitter(
            image_getter=copy.copy(self.__image_getter),
            corner_getter=copy.copy(self.__corner_detector),
            inventory_detector=copy.copy(self.__inventory_detector)
        )

    def set_image_getter(self, image_getter: ImageGetter):
        self.__image_getter = image_getter

    def set_corner_detector(self, corner_detector: CornerDetector):
        self.__corner_detector = corner_detector

    def set_inventory_detector(self, inventory_detector: InventoryDetector | None):
        self.__inventory_detector = inventory_detector

    def get_image_getter(self):
        return self.__image_getter

    def get_corner_detector(self):
        return self.__corner_detector

    def get_inventory_detector(self):
        return self.__inventory_detector
