import copy

import numpy as np
import shogi

from extra.figures import Figure, Direction, get_figure_image
from extra.types import FigureBoard, DirectionBoard, ImageNP, Inventory
from dataclasses import dataclass
from extra import utils
import cv2


@dataclass(frozen=True)
class Board:
    figures: FigureBoard
    directions: DirectionBoard
    inventory_black: Inventory = None
    inventory_white: Inventory = None

    def to_str_figures(self):
        return utils.board_to_str(self.figures)

    def to_str_directions(self):
        return utils.board_to_str(self.directions)

    def to_str(self) -> str:
        s = "Figures:     Directions:\n"
        for fig_row, dir_row in zip(self.figures, self.directions):
            s += "".join([cell.value for cell in fig_row])
            s += "    "
            s += "".join([cell.value for cell in dir_row])
            s += "\n"
        return s

    def to_image(self) -> ImageNP:
        BOARD_SIZE = 1000
        FIGURE_SIZE = BOARD_SIZE // 9
        INVENTORY_FIGURE_SIZE = FIGURE_SIZE
        INVENTORY_MARGIN = 50

        board_img = np.full((BOARD_SIZE, BOARD_SIZE, 3), [255, 255, 255], dtype=np.uint8)

        # Adding figures icons
        figure_step = BOARD_SIZE // 9
        for i in range(9):
            for j in range(9):
                y = figure_step * i
                x = figure_step * j
                figure = self.figures[i][j]
                direction = self.directions[i][j]
                if figure != Figure.EMPTY:
                    figure_icon = get_figure_image(figure, direction)
                    figure_icon = cv2.resize(figure_icon, (FIGURE_SIZE, FIGURE_SIZE))
                    utils.overlay_image_on_image(board_img, figure_icon, x, y)

        # Drawing grid
        grid_step = BOARD_SIZE // 9
        for i in range(10):
            y = i * grid_step
            cv2.line(board_img, (0, y), (BOARD_SIZE, y), 0, thickness=5)
        for j in range(10):
            x = j * grid_step
            cv2.line(board_img, (x, 0), (x, BOARD_SIZE), 0, thickness=5)

        # Drawing inventories
        black_inv, white_inv = self.get_inventory_lists()
        margin_line = np.full((INVENTORY_MARGIN, BOARD_SIZE, 3), [255, 255, 255], dtype=np.uint8)
        black_inv_line = np.full((INVENTORY_FIGURE_SIZE, BOARD_SIZE, 3), [255, 255, 255], dtype=np.uint8)
        white_inv_line = np.full((INVENTORY_FIGURE_SIZE, BOARD_SIZE, 3), [255, 255, 255], dtype=np.uint8)
        for i, black_inv_fig in enumerate(black_inv):
            figure_icon = get_figure_image(black_inv_fig, Direction.UP)
            figure_icon = cv2.resize(figure_icon, (INVENTORY_FIGURE_SIZE, INVENTORY_FIGURE_SIZE))
            utils.overlay_image_on_image(black_inv_line, figure_icon,
                                         x=BOARD_SIZE - INVENTORY_FIGURE_SIZE * (i + 1),
                                         y=0)
        for i, white_inv_fig in enumerate(white_inv):
            figure_icon = get_figure_image(white_inv_fig, Direction.DOWN)
            figure_icon = cv2.resize(figure_icon, (INVENTORY_FIGURE_SIZE, INVENTORY_FIGURE_SIZE))
            utils.overlay_image_on_image(white_inv_line, figure_icon,
                                         x=INVENTORY_FIGURE_SIZE * i,
                                         y=0)
        board_img = np.array([
            *white_inv_line,
            *margin_line,
            *board_img,
            *margin_line,
            *black_inv_line
        ])
        return board_img

    def to_shogi_board(self) -> shogi.Board:
        board = shogi.Board()
        board.clear()

        to_shogi_type = {
            Figure.KING: shogi.KING,
            Figure.PAWN: shogi.PAWN,
            Figure.LANCE: shogi.LANCE,
            Figure.KNIGHT: shogi.KNIGHT,
            Figure.SILVER: shogi.SILVER,
            Figure.GOLD: shogi.GOLD,
            Figure.BISHOP: shogi.BISHOP,
            Figure.ROOK: shogi.ROOK,
            Figure.PAWN_PROM: shogi.PROM_PAWN,
            Figure.LANCE_PROM: shogi.PROM_LANCE,
            Figure.KNIGHT_PROM: shogi.PROM_KNIGHT,
            Figure.SILVER_PROM: shogi.PROM_SILVER,
            Figure.BISHOP_PROM: shogi.PROM_BISHOP,
            Figure.ROOK_PROM: shogi.PROM_ROOK,
        }

        to_shogi_color = {
            Direction.DOWN: shogi.WHITE,
            Direction.UP: shogi.BLACK,
        }

        # Adding pieces on board
        for i in range(9):
            for j in range(9):
                square = shogi.SQUARES[i * 9 + j]
                figure = self.figures[i][j]
                direction = self.directions[i][j]
                if figure != Figure.EMPTY and direction != Direction.NONE:
                    piece_type = to_shogi_type[figure]
                    color = to_shogi_color[direction]
                    piece = shogi.Piece(piece_type, color)
                    board.set_piece_at(square, piece)

        # Adding pieces from inventory
        if self.inventory_black and self.inventory_white:
            for figure in Figure:
                if figure.is_droppable():
                    count_black = self.inventory_black[figure]
                    count_white = self.inventory_white[figure]
                    piece_type = to_shogi_type[figure]
                    board.add_piece_into_hand(piece_type, shogi.BLACK, count_black)
                    board.add_piece_into_hand(piece_type, shogi.WHITE, count_white)
        return board

    @classmethod
    def get_empty_board(cls):
        figures = [[Figure.EMPTY] * 9 for _ in range(9)]
        directions = [[Direction.NONE] * 9 for _ in range(9)]
        return Board(figures, directions)

    @classmethod
    def default_board(cls):
        row_main = lambda: [Figure.LANCE, Figure.KNIGHT, Figure.SILVER, Figure.GOLD, Figure.KING, Figure.GOLD, Figure.SILVER, Figure.KNIGHT, Figure.LANCE]
        row_pawn = lambda: [Figure.PAWN] * 9
        row_empty = lambda: [Figure.EMPTY] * 9
        figures = [
            row_main(),
            [Figure.EMPTY, Figure.ROOK, *[Figure.EMPTY] * 5, Figure.BISHOP, Figure.EMPTY],
            row_pawn(),
            row_empty(),
            row_empty(),
            row_empty(),
            row_pawn(),
            [Figure.EMPTY, Figure.BISHOP, *[Figure.EMPTY] * 5, Figure.ROOK, Figure.EMPTY],
            row_main(),
        ]
        directions = [
            [Direction.DOWN] * 9,
            [Direction.NONE, Direction.DOWN, *[Direction.NONE] * 5, Direction.DOWN, Direction.NONE],
            [Direction.DOWN] * 9,
            [Direction.NONE] * 9,
            [Direction.NONE] * 9,
            [Direction.NONE] * 9,
            [Direction.UP] * 9,
            [Direction.NONE, Direction.UP, *[Direction.NONE] * 5, Direction.UP, Direction.NONE],
            [Direction.UP] * 9,
        ]
        return Board(figures, directions)

    def get_inventory_lists(self) -> tuple[list[Figure], list[Figure]]:
        black = []
        white = []
        if self.inventory_black is not None:
            for figure in self.inventory_black:
                count = self.inventory_black[figure]
                black += [figure] * count
        if self.inventory_white is not None:
            for figure in self.inventory_white:
                count = self.inventory_white[figure]
                white += [figure] * count
        return black, white

    def lishogi_link(self) -> str:
        sfen = self.to_shogi_board().sfen()
        sfen = sfen.replace(" ", "_")
        url = f"https://lishogi.org/editor/{sfen}"
        return url

    def __str__(self):
        sfen = self.to_shogi_board().sfen()
        return sfen

    def copy(self):
        return Board(
            copy.deepcopy(self.figures),
            copy.deepcopy(self.directions),
            copy.deepcopy(self.inventory_black),
            copy.deepcopy(self.inventory_white),
        )

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return hash(self) == hash(other)