from extra.types import FigureBoard, DirectionBoard
from Elements.Board.Move import *
from ..Board.Board import Board, BoardChangeStatus


class BoardMemorizerGreedy:
    __move_history: list[Move]

    __confirmed_board: Board  # board that is considered true

    # proposed boards that come after confirmed board
    # there may be several proposed boards because of noise
    # if newly proposed board matches with one in this list, then board from list becomes confirmed
    __proposed_boards: list[Board]

    update_status: BoardChangeStatus = BoardChangeStatus.VALID_MOVE

    def __init__(self):
        self.__move_history = []
        self.__confirmed_board = Board.default_board()
        self.__proposed_boards = []

    def update(self, figures: FigureBoard, directions: DirectionBoard, certainty_score: float) -> None:
        """Updates board and stores status of update in 'update_status' variable"""
        new_board = Board(figures, directions)
        if certainty_score < 0.99:
            self.update_status = BoardChangeStatus.LOW_CERTAINTY
            return

        # Checking if new board is continuation of some of the proposed ones
        for board in self.__proposed_boards:
            change_status = board.get_change_status(new_board)
            match change_status:
                case BoardChangeStatus.NOTHING_CHANGED:
                    self.update_status = BoardChangeStatus.NOTHING_CHANGED
                    return
                case BoardChangeStatus.VALID_MOVE:
                    self.__confirmed_board = board
                    self.__proposed_boards = [new_board]
                    move = board.get_move(new_board)
                    self.__move_history.append(move)
                    self.update_status = BoardChangeStatus.VALID_MOVE
                    return

        # Could not continue any proposed board. Trying to match with confirmed board
        change_status = self.__confirmed_board.get_change_status(new_board)
        self.update_status = change_status
        if change_status == BoardChangeStatus.VALID_MOVE:
            self.__proposed_boards.append(new_board)

    def get_board(self) -> Board:
        return self.__confirmed_board

    def get_kif(self) -> str:
        s = """手合割：平手
先手：
後手：
手数----指手----消費時間--
"""

        for i, move in enumerate(self.__move_history):
            signature = move.to_kif()
            row_fmt = "{:>4} {}\n"
            s += row_fmt.format(i + 1, signature)

        return s

    def clear(self):
        self.__move_history.clear()
        self.__proposed_boards.clear()
        self.__confirmed_board = Board.default_board()

    def get_moves(self) -> list[Move]:
        return self.__move_history
