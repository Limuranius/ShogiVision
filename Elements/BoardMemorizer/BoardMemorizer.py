import shogi

from extra.types import FigureBoard, DirectionBoard
from .BoardCounter import BoardCounter
from .Move import *
from ..Board import Board, BoardChangeStatus


class BoardMemorizer:
    __boards_counter: BoardCounter
    __move_history: list[Move]
    __board: shogi.Board

    # True means that player on the lower half of the board moved first
    # If None then side will be decided automatically
    lower_moves_first: bool | None

    update_status: BoardChangeStatus = BoardChangeStatus.VALID_MOVE

    def __init__(self, lower_moves_first: bool = None, buffer_size: int = 10):
        self.__boards_counter = BoardCounter(buffer_size=buffer_size)
        self.__move_history = []
        self.lower_moves_first = lower_moves_first
        self.__board = shogi.Board()

    def update(self, figures: FigureBoard, directions: DirectionBoard, certainty_score: float) -> None:
        """Updates board and stores status of update in 'update_status' variable"""
        new_board = Board(figures, directions)
        if certainty_score < 0.99:
            change_status = BoardChangeStatus.LOW_CERTAINTY
        else:
            change_status = self.__get_change_status(new_board)
        self.update_status = change_status
        match change_status:
            case BoardChangeStatus.NOTHING_CHANGED:
                self.__boards_counter.update(new_board)
            case BoardChangeStatus.ACCUMULATING_DATA:
                self.__boards_counter.update(new_board)
            case BoardChangeStatus.INVALID_MOVE | BoardChangeStatus.ILLEGAL_MOVE:
                pass
            case BoardChangeStatus.VALID_MOVE:
                curr_board = self.__boards_counter.get_max_board()
                self.__boards_counter.update(new_board)
                new_curr_board = self.__boards_counter.get_max_board()
                if curr_board != new_curr_board:
                    move = get_move(curr_board, new_curr_board)
                    if self.lower_moves_first is None:
                        # need to infer side based on first move
                        x, y = move.origin
                        direction = curr_board.directions[y - 1][x - 1]
                        self.lower_moves_first = direction == Direction.UP
                    self.__move_history.append(move)
                    self.__board.push_usi(
                        move.apply_side_transformation(self.lower_moves_first).to_usi()
                    )
            case BoardChangeStatus.LOW_CERTAINTY:
                pass

    def get_board(self) -> Board:
        return self.__boards_counter.get_max_board()

    def get_kif(self) -> str:
        s = """手合割：平手
先手：
後手：
手数----指手----消費時間--
"""

        for i, move in enumerate(self.__move_history):
            signature = move.apply_side_transformation(self.lower_moves_first).to_kif()
            row_fmt = "{:>4} {}\n"
            s += row_fmt.format(i + 1, signature)

        return s

    def __remake_board(self):
        self.__board = shogi.Board()
        for move in self.__move_history:
            self.__board.push_usi(
                move.apply_side_transformation(self.lower_moves_first).to_usi()
            )

    def set_side(self, lower_moves_first: bool):
        self.lower_moves_first = lower_moves_first
        self.__remake_board()

    def __get_change_status(self, new_board: Board) -> BoardChangeStatus:
        if not self.__boards_counter.filled:
            return BoardChangeStatus.ACCUMULATING_DATA
        curr_board = self.__boards_counter.get_max_board()
        if new_board == curr_board:
            return BoardChangeStatus.NOTHING_CHANGED
        move = get_move(
            curr_board,
            new_board
        )
        if move is None:
            return BoardChangeStatus.INVALID_MOVE
        if self.lower_moves_first is None:  # Don't know side yet
            # Trying both sides
            usi1 = move.apply_side_transformation(True).to_usi()
            usi2 = move.apply_side_transformation(False).to_usi()
            is_move1_legal = self.__board.is_legal(shogi.Move.from_usi(usi1))
            is_move2_legal = self.__board.is_legal(shogi.Move.from_usi(usi2))
            if is_move1_legal or is_move2_legal:
                return BoardChangeStatus.VALID_MOVE
            else:
                return BoardChangeStatus.ILLEGAL_MOVE
        usi = move.apply_side_transformation(self.lower_moves_first).to_usi()
        is_move_legal = self.__board.is_legal(shogi.Move.from_usi(usi))
        if is_move_legal:
            return BoardChangeStatus.VALID_MOVE
        else:
            return BoardChangeStatus.ILLEGAL_MOVE

    def clear(self):
        self.__boards_counter.clear()
        self.__move_history.clear()
        self.__board = shogi.Board()

    def get_moves(self) -> list[Move]:
        return self.__move_history
