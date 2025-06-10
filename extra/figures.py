from __future__ import annotations
from enum import Enum

import numpy as np

from config import paths
import os
import cv2


class Figure(Enum):
    PAWN = "p"
    KING = "K"
    LANCE = "l"
    KNIGHT = "n"
    SILVER = "s"
    GOLD = "G"
    BISHOP = "b"
    ROOK = "r"
    EMPTY = "."

    PAWN_PROM = "P"
    LANCE_PROM = "L"
    KNIGHT_PROM = "N"
    SILVER_PROM = "S"
    BISHOP_PROM = "B"
    ROOK_PROM = "R"

    def to_jp(self) -> str:
        translate_table = {
            self.PAWN: "歩",
            self.KING: "玉",
            self.LANCE: "香",
            self.KNIGHT: "桂",
            self.SILVER: "銀",
            self.GOLD: "金",
            self.BISHOP: "角",
            self.ROOK: "飛",

            self.PAWN_PROM: "と",
            self.LANCE_PROM: "成香",
            self.KNIGHT_PROM: "成桂",
            self.SILVER_PROM: "成銀",
            self.BISHOP_PROM: "馬",
            self.ROOK_PROM: "龍",
        }
        return translate_table[self]

    def promoted(self) -> Figure:
        return promotion_table[self]

    def unpromoted(self) -> Figure:
        return unpromotion_table[self]

    def is_promotable(self) -> bool:
        return self in promotion_table

    def is_promoted(self) -> bool:
        return self in promotion_table.values()

    def is_droppable(self) -> bool:
        return self in droppable

    def get_moves(self, direction: Direction) -> np.ndarray:
        # Get moves of figure in array of (dy, dx) pairs
        if direction == Direction.UP:
            return figure_moves[self]
        return inv_figure_moves[self]


# Moves
king_moves = {(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)}
rook_moves = {
    *[(0, dx) for dx in range(-8, 9) if dx != 0],
    *[(dy, 0) for dy in range(-8, 9) if dy != 0]
}
bishop_moves = {
    *[(d, d) for d in range(-8, 9) if d != 0],
    *[(d, -d) for d in range(-8, 9) if d != 0],
}
gold_moves = {(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)}
figure_moves = {
    Figure.PAWN: {(-1, 0)},
    Figure.KING: king_moves,
    Figure.LANCE: {(-dy, 0) for dy in range(1, 9)},
    Figure.KNIGHT: {(-2, -1), (-2, 1)},
    Figure.SILVER: {(-1, -1), (-1, 0), (1, -1), (-1, 1), (1, 1)},
    Figure.GOLD: gold_moves,
    Figure.BISHOP: bishop_moves,
    Figure.ROOK: rook_moves,
    Figure.PAWN_PROM: gold_moves,
    Figure.LANCE_PROM: gold_moves,
    Figure.KNIGHT_PROM: gold_moves,
    Figure.SILVER_PROM: gold_moves,
    Figure.BISHOP_PROM: king_moves | bishop_moves,
    Figure.ROOK_PROM: king_moves | rook_moves,
}
inv_figure_moves = {fig: {(-dy, dx) for dy, dx in figure_moves[fig]} for fig in figure_moves}
figure_moves = {fig: np.array(list(figure_moves[fig])) for fig in figure_moves}
inv_figure_moves = {fig: np.array(list(inv_figure_moves[fig])) for fig in inv_figure_moves}

promotion_table = {
    Figure.PAWN: Figure.PAWN_PROM,
    Figure.LANCE: Figure.LANCE_PROM,
    Figure.KNIGHT: Figure.KNIGHT_PROM,
    Figure.SILVER: Figure.SILVER_PROM,
    Figure.BISHOP: Figure.BISHOP_PROM,
    Figure.ROOK: Figure.ROOK_PROM,
}

unpromotion_table = {
    Figure.PAWN_PROM: Figure.PAWN,
    Figure.LANCE_PROM: Figure.LANCE,
    Figure.KNIGHT_PROM: Figure.KNIGHT,
    Figure.SILVER_PROM: Figure.SILVER,
    Figure.BISHOP_PROM: Figure.BISHOP,
    Figure.ROOK_PROM: Figure.ROOK,
}

droppable = {
    Figure.PAWN,
    Figure.LANCE,
    Figure.KNIGHT,
    Figure.SILVER,
    Figure.BISHOP,
    Figure.ROOK,
    Figure.GOLD
}


class Direction(Enum):
    UP = "U"
    DOWN = "D"
    NONE = "."

    def opposite(self):
        if self == Direction.UP:
            return Direction.DOWN
        return Direction.UP

    def promotion_zone_array_y(self) -> list[int]:
        return {
            Direction.UP: [0, 1, 2],
            Direction.DOWN: [6, 7, 8],
        }[self]



FIGURE_ICONS_PATHS = {
    Figure.PAWN: os.path.join(paths.FIGURE_ICONS_DIR, "pawn.png"),
    Figure.BISHOP: os.path.join(paths.FIGURE_ICONS_DIR, "bishop.png"),
    Figure.ROOK: os.path.join(paths.FIGURE_ICONS_DIR, "rook.png"),
    Figure.LANCE: os.path.join(paths.FIGURE_ICONS_DIR, "lance.png"),
    Figure.KNIGHT: os.path.join(paths.FIGURE_ICONS_DIR, "knight.png"),
    Figure.SILVER: os.path.join(paths.FIGURE_ICONS_DIR, "silver.png"),
    Figure.GOLD: os.path.join(paths.FIGURE_ICONS_DIR, "gold.png"),
    Figure.KING: os.path.join(paths.FIGURE_ICONS_DIR, "king.png"),
    Figure.EMPTY: os.path.join(paths.FIGURE_ICONS_DIR, "empty.png"),

    Figure.PAWN_PROM: os.path.join(paths.FIGURE_ICONS_DIR, "promoted pawn.png"),
    Figure.BISHOP_PROM: os.path.join(paths.FIGURE_ICONS_DIR, "promoted bishop.png"),
    Figure.ROOK_PROM: os.path.join(paths.FIGURE_ICONS_DIR, "promoted rook.png"),
    Figure.LANCE_PROM: os.path.join(paths.FIGURE_ICONS_DIR, "promoted lance.png"),
    Figure.KNIGHT_PROM: os.path.join(paths.FIGURE_ICONS_DIR, "promoted knight.png"),
    Figure.SILVER_PROM: os.path.join(paths.FIGURE_ICONS_DIR, "promoted silver.png"),
}


def get_figure_image(figure: Figure, direction: Direction):
    figure_img = cv2.imread(FIGURE_ICONS_PATHS[figure])
    if direction == Direction.DOWN:
        figure_img = cv2.rotate(figure_img, cv2.ROTATE_180)
    return figure_img
