# For type annotations to avoid circular imports
from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from Elements.Board import Board

import cv2
import numpy as np

from Elements.Board.Move import Move
from extra import utils
from extra.figures import get_figure_image, Direction, Figure
from extra.types import ImageNP


class BoardDrawer:
    board: Board

    show_inventory: bool
    show_arrows: bool
    show_turn_margin: bool

    board_size: int  # width and height of board
    figure_size: int
    inventory_figure_size: int
    inventory_margin_size: int

    moves: list[Move]

    def __init__(
            self,
            board: Board,
            show_inventory=True,
            show_arrows=True,
            show_turn=True,
    ):
        self.board = board
        self.board_size = 1000
        self.figure_size = self.board_size // 9
        self.inventory_figure_size = self.figure_size
        self.inventory_margin_size = 50

        self.show_inventory = show_inventory
        self.show_arrows = show_arrows
        self.show_turn_margin = show_turn

        self.moves = []

    def __build_board_image(self) -> ImageNP:
        board_img = np.full(
            (self.board_size, self.board_size, 3),
            [255, 255, 255],
            dtype=np.uint8
        )

        # Adding figures icons
        for i in range(9):
            for j in range(9):
                y = self.figure_size * i
                x = self.figure_size * j
                figure = self.board.figures[i][j]
                direction = self.board.directions[i][j]
                if figure != Figure.EMPTY:
                    figure_icon = get_figure_image(figure, direction)
                    figure_icon = cv2.resize(figure_icon, (self.figure_size, self.figure_size))
                    utils.overlay_image_on_image(board_img, figure_icon, x, y)

        # Drawing grid
        grid_step = self.board_size // 9
        for i in range(10):
            y = i * grid_step
            cv2.line(board_img, (0, y), (self.board_size, y), 0, thickness=5)
        for j in range(10):
            x = j * grid_step
            cv2.line(board_img, (x, 0), (x, self.board_size), 0, thickness=5)
        return board_img

    def __build_inventory_images(self) -> tuple[ImageNP, ImageNP]:
        # Drawing inventories

        black_inv_line = np.full((self.inventory_figure_size, self.board_size, 3), [255, 255, 255], dtype=np.uint8)
        white_inv_line = np.full((self.inventory_figure_size, self.board_size, 3), [255, 255, 255], dtype=np.uint8)
        black_inv, white_inv = self.board.inventory_black, self.board.inventory_white
        if black_inv is None:
            black_inv = dict()
        if white_inv is None:
            white_inv = dict()
        for i, (black_inv_fig, count) in enumerate(black_inv.items()):
            x = self.board_size - self.inventory_figure_size * (i + 1)
            y = 0
            figure_icon = get_figure_image(black_inv_fig, Direction.UP)
            figure_icon = cv2.resize(figure_icon, (self.inventory_figure_size, self.inventory_figure_size))
            utils.overlay_image_on_image(
                black_inv_line,
                figure_icon,
                x=x,
                y=y,
            )
            cv2.putText(black_inv_line, str(count) if count > 1 else "",
                        (x + self.inventory_figure_size // 2, y + self.inventory_figure_size),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4,
                        lineType=cv2.LINE_AA)
        for i, (white_inv_fig, count) in enumerate(white_inv.items()):
            x = self.inventory_figure_size * i
            y = 0
            figure_icon = get_figure_image(white_inv_fig, Direction.DOWN)
            figure_icon = cv2.resize(figure_icon, (self.inventory_figure_size, self.inventory_figure_size))
            utils.overlay_image_on_image(
                white_inv_line,
                figure_icon,
                x=x,
                y=y,
            )
            cv2.putText(white_inv_line, str(count) if count > 1 else "",
                        (x + self.inventory_figure_size // 2, y + self.inventory_figure_size),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4,
                        lineType=cv2.LINE_AA)
        return black_inv_line, white_inv_line

    def __build_margin_lines(self) -> tuple[ImageNP, ImageNP]:
        # Adding margins to inventories and coloring them based on whose turn is it now
        if self.board.turn == Direction.UP:
            margin_line_black = np.full((self.inventory_margin_size, self.board_size, 3), [0, 255, 0], dtype=np.uint8)
            margin_line_white = np.full((self.inventory_margin_size, self.board_size, 3), [255, 255, 255],
                                        dtype=np.uint8)
        else:
            margin_line_black = np.full((self.inventory_margin_size, self.board_size, 3), [255, 255, 255],
                                        dtype=np.uint8)
            margin_line_white = np.full((self.inventory_margin_size, self.board_size, 3), [0, 255, 0], dtype=np.uint8)
        return margin_line_black, margin_line_white

    def to_image(self) -> ImageNP:
        board_img = self.__build_board_image()

        if self.show_inventory:
            black_inv_line, white_inv_line = self.__build_inventory_images()
        else:
            black_inv_line, white_inv_line = [], []

        if self.show_turn_margin:
            margin_line_black, margin_line_white = self.__build_margin_lines()
        else:
            margin_line_black, margin_line_white = [], []

        board_img = np.array([
            *white_inv_line,
            *margin_line_white,
            *board_img,
            *margin_line_black,
            *black_inv_line
        ])

        if self.show_arrows:
            for i, move in enumerate(self.moves):
                self.__draw_move(board_img, move, str(i + 1))

        return board_img

    def __draw_move(self, image: ImageNP, move: Move, text: str) -> None:
        x_end, y_end = self.__get_board_cell_coord(move.array_destination[0], move.array_destination[1])
        if move.is_drop:
            x_start, y_start = self.__get_inventory_figure_coord(move.figure, move.direction)
        else:
            x_start, y_start = self.__get_board_cell_coord(move.array_origin[0], move.array_origin[1])
        self.__draw_arrow_with_text(
            image,
            (x_start, y_start),
            (x_end, y_end),
            text,
            is_promotion=move.is_promotion,
        )

    def __draw_arrow_with_text(
            self,
            image: ImageNP,
            start: tuple[int, int],  # (x, y)
            end: tuple[int, int],  # (x, y)
            text: str,
            color=(0, 150, 0),
            thickness=5,

            is_promotion: bool = False,
            plus_size: int = 10,
            plus_thickness: int = 2,
            plus_color=(150, 0, 0),
    ) -> None:
        """Draws arrow from :start to :end and puts text in the middle"""
        cv2.arrowedLine(image, start, end, color, thickness,
                        line_type=cv2.LINE_AA, tipLength=0.1)

        middle_x = (start[0] + end[0]) // 2
        middle_y = (start[1] + end[1]) // 2

        if is_promotion:
            # Drawing plus sign
            cv2.line(image,
                     (middle_x - plus_size, middle_y),  # Left point
                     (middle_x + plus_size, middle_y),  # Right point
                     plus_color, plus_thickness, cv2.LINE_AA)
            cv2.line(image,
                     (middle_x, middle_y - plus_size),  # Top point
                     (middle_x, middle_y + plus_size),  # Bottom point
                     plus_color, plus_thickness, cv2.LINE_AA)

        cv2.putText(image, text,
                    (middle_x, middle_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5,
                    lineType=cv2.LINE_AA)

    def __get_inventory_figure_coord(self, figure: Figure, direction: Direction) -> tuple[int, int]:
        """Returns pixel coordinate of figure in inventory"""
        assert self.show_inventory == True
        inv_i = {Direction.UP: 0, Direction.DOWN: 1}[direction]
        inv = list(self.board.inventories[inv_i].keys())
        figure_i = inv.index(figure)
        if direction == Direction.UP:
            cy = (
                    self.inventory_figure_size
                    + self.inventory_margin_size * self.show_turn_margin
                    + self.board_size
                    + self.inventory_margin_size * self.show_turn_margin
                    + self.inventory_figure_size // 2
            )
            cx = self.board_size - self.inventory_figure_size * figure_i - self.inventory_figure_size // 2
        else:
            cy = self.inventory_figure_size // 2
            cx = self.inventory_figure_size * figure_i + self.inventory_figure_size // 2
        return cx, cy

    def __get_board_cell_coord(self, i: int, j: int) -> tuple[int, int]:
        """Converts board coordinate to pixel coordinate"""
        y_margin = self.show_inventory * self.inventory_figure_size + self.show_turn_margin * self.inventory_margin_size
        cy = y_margin + self.figure_size * i + self.figure_size // 2
        cx = self.figure_size * j + self.figure_size // 2
        return cx, cy

    def add_move(self, move: Move):
        self.moves.append(move)

    def add_moves(self, moves: list[Move]):
        self.moves += moves

    def augment_real_image(self, real_image: ImageNP) -> ImageNP:
        pass

    def get_bounding_boxes(self) -> dict:
        """
        Returns bounding boxes (x, y, w, h) of each point of interest on board image. This includes:
            - each cell on board
            - figures in inventory if present
        """
        ans = dict()
        ans["board"] = [[None] * 9 for _ in range(9)]
        y_margin = self.show_inventory * self.inventory_figure_size + self.show_turn_margin * self.inventory_margin_size
        for i in range(9):
            for j in range(9):
                x = self.figure_size * j
                y = y_margin + self.figure_size * i
                w = self.figure_size
                h = self.figure_size
                ans["board"][i][j] = (x, y, w, h)

        if self.show_inventory:
            ans["inventory_black"] = dict()
            ans["inventory_white"] = dict()
            w = h = self.inventory_figure_size
            for i, figure in enumerate(self.board.inventory_black.keys()):
                y = (self.inventory_figure_size
                     + 2 * self.inventory_margin_size * self.show_turn_margin
                     + self.board_size)
                x = self.board_size - self.inventory_figure_size * (i + 1)
                ans["inventory_black"][figure] = (x, y, w, h)
            for i, figure in enumerate(self.board.inventory_white.keys()):
                y = 0
                x = self.inventory_figure_size * i
                ans["inventory_white"][figure] = (x, y, w, h)

        return ans