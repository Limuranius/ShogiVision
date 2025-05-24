from extra.types import FigureBoard, DirectionBoard
from .Move import *
from ..Board import Board, BoardChangeStatus
from .BoardsTree import BoardsTree


class BoardMemorizerTree:
    _tree: BoardsTree

    update_status: BoardChangeStatus = BoardChangeStatus.VALID_MOVE

    def __init__(self):
        self._tree = BoardsTree()
        self._tree.insert(Board.default_board())

    def update(self, figures: FigureBoard, directions: DirectionBoard, certainty_score: float) -> None:
        """Updates board and stores status of update in 'update_status' variable"""
        new_board = Board(figures, directions)
        if certainty_score < 0.99:
            self.update_status = BoardChangeStatus.LOW_CERTAINTY
            return

        status = self._tree.insert(new_board)
        self.update_status = status

    def get_board(self) -> Board:
        return self._tree.get_last_boards()[0]

    def get_kif(self) -> str:
        s = """手合割：平手
先手：
後手：
手数----指手----消費時間--
"""

        for i, move in enumerate(self.get_moves()):
            signature = move.to_kif()
            row_fmt = "{:>4} {}\n"
            s += row_fmt.format(i + 1, signature)

        return s

    def clear(self):
        self._tree.clear()

    def get_moves(self) -> list[Move]:
        return self._tree.move_histories()[0]

    def fill_missing_boards(self, target_board: Board, max_turns=1):
        """Bruteforces boards until they match with :target_board"""
        raise NotImplementedError()
