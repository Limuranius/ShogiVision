import copy

import shogi

from extra.types import FigureBoard, DirectionBoard
from .BoardChangeStatus import BoardChangeStatus
from .Move import *
from .board_utils import get_move
from ..Board import Board


class BoardMemorizerGreedy:
    __move_history: list[Move]
    __board: shogi.Board
    __last_board: Board

    # True means that player on the lower half of the board moved first
    # If None then side will be decided automatically
    lower_moves_first: bool | None

    update_status: BoardChangeStatus = BoardChangeStatus.VALID_MOVE

    def __init__(self, lower_moves_first: bool = None):
        self.__move_history = []
        self.lower_moves_first = lower_moves_first
        self.__board = shogi.Board()
        self.__last_board = Board.default_board()

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
                pass
            case BoardChangeStatus.ACCUMULATING_DATA:
                pass
            case BoardChangeStatus.INVALID_MOVE | BoardChangeStatus.ILLEGAL_MOVE:
                pass
            case BoardChangeStatus.VALID_MOVE:
                move = get_move(self.__last_board, new_board)
                self.__last_board = new_board
                self.__move_history.append(move)
                self.__board.push_usi(
                    move.apply_side_transformation(self.lower_moves_first).to_usi()
                )
            case BoardChangeStatus.LOW_CERTAINTY:
                pass

    def get_board(self) -> Board:
        return self.__last_board

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
        curr_board = self.__last_board
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
            if is_move1_legal:
                self.lower_moves_first = True
                return BoardChangeStatus.VALID_MOVE
            elif is_move2_legal:
                self.lower_moves_first = False
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
        self.__move_history.clear()
        self.__board = shogi.Board()
        self.__last_board = Board.default_board()

    def get_moves(self) -> list[Move]:
        return self.__move_history

    def copy(self):
        memorizer = BoardMemorizerGreedy()
        memorizer.__board = copy.deepcopy(self.__board)
        memorizer.__move_history = copy.deepcopy(self.__move_history)
        memorizer.__last_board = self.__last_board
        memorizer.update_status = self.update_status
        memorizer.lower_moves_first = self.lower_moves_first
        return memorizer

    def is_sequence_valid(self, boards: list[Board]) -> bool:
        """
        From current state of memorizer checks if sequence of boards is a valid continuation
        This means that if any board produces invalid or illegal move then sequence is considered invalid
        Does not update memorizer state
        """
        mem = self.copy()  # will feed boards to copy of current memorizer
        for board in boards:
            mem.update(board.figures, board.directions, 1.0)
            if mem.update_status in [BoardChangeStatus.INVALID_MOVE, BoardChangeStatus.ILLEGAL_MOVE]:
                return False
        return True
