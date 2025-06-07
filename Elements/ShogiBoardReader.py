import copy
from collections import defaultdict
from Elements.Recognizers import Recognizer
from extra.figures import Figure
from .BoardMemorizer import BoardMemorizer
from .BoardSplitter import BoardSplitter
from extra.types import ImageNP, FigureBoard, DirectionBoard, Inventory
from .Board.Board import Board, BoardChangeStatus


class ShogiBoardReader:
    __board_splitter: BoardSplitter
    __recognizer: Recognizer
    __memorizer: BoardMemorizer

    __figures: FigureBoard
    __directions: DirectionBoard

    confidence_score: float  # How confident last recognition was

    def __init__(
            self,
            board_splitter: BoardSplitter,
            recognizer: Recognizer,
            memorizer: BoardMemorizer = None):
        self.__board_splitter = board_splitter
        self.__recognizer = recognizer
        self.__memorizer = memorizer

        empty_board = Board.get_empty_board()
        self.__figures = empty_board.figures
        self.__directions = empty_board.directions

    def get_full_img(
            self,
            show_borders: bool = False,
            show_grid: bool = False,
            show_inventories: bool = False,
    ) -> ImageNP:
        """
        Returns original image of board
        Can also draw borders of board and grid
        """
        return self.__board_splitter.get_full_img(show_borders, show_grid, show_inventories)

    def get_board_image_no_perspective(self,
                                       show_grid: bool = False) -> ImageNP:
        """Returns image of board without perspective and surroundings"""
        return self.__board_splitter.get_board_image_no_perspective(show_grid)

    def update(self) -> None:
        """
        Runs recognizer and memorizer to update state of board
        """
        cells = self.__board_splitter.get_board_cells()
        figures, directions, score = self.__recognizer.recognize_board(cells)

        self.__figures = figures
        self.__directions = directions
        self.confidence_score = score

        if self.__memorizer is not None:
            self.__memorizer.update(self.__figures, self.__directions, score)

    def get_last_update_status(self) -> BoardChangeStatus:
        if self.__memorizer is not None:
            return self.__memorizer.update_status
        else:
            return BoardChangeStatus.VALID_MOVE

    def get_board(self) -> Board:
        if self.__memorizer is None:
            if self.__board_splitter.get_inventory_detector() is None:
                return Board(self.__figures, self.__directions)
            else:
                inventory_black, inventory_white = self.get_inventories()
                return Board(self.__figures, self.__directions, inventory_black, inventory_white)
        else:
            return self.__memorizer.get_board()

    def set_board_splitter(self, splitter: BoardSplitter) -> None:
        self.__board_splitter = splitter

    def set_recognizer(self, recognizer: Recognizer) -> None:
        self.__recognizer = recognizer

    def set_memorizer(self, memorizer: BoardMemorizer) -> None:
        self.__memorizer = memorizer

    def get_board_splitter(self) -> BoardSplitter:
        return self.__board_splitter

    def get_recognizer(self) -> Recognizer:
        return self.__recognizer

    def get_memorizer(self) -> BoardMemorizer:
        return self.__memorizer

    def get_inventories(self) -> tuple[Inventory, Inventory]:
        i1_imgs, i2_imgs = self.__board_splitter.get_inventory_cells()
        i1 = defaultdict(int)
        i2 = defaultdict(int)
        for img in i1_imgs:
            figure, direction = self.__recognizer.recognize_cell(img)
            if figure != Figure.EMPTY:
                i1[figure] += 1
        for img in i2_imgs:
            figure, direction = self.__recognizer.recognize_cell(img)
            if figure != Figure.EMPTY:
                i2[figure] += 1
        return i1, i2

    def get_kif(self) -> str:
        if self.__memorizer is None:
            return ""
        else:
            return self.__memorizer.get_kif()

    def __copy__(self):
        return ShogiBoardReader(
            board_splitter=copy.copy(self.__board_splitter),
            recognizer=copy.copy(self.__recognizer),
            memorizer=copy.copy(self.__memorizer)
        )
