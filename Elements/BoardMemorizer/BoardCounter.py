from collections import defaultdict
from Elements.Board import Board


class BoardCounter:
    __memorize_count: int  # How many frames of board are memorized and calculated statistics on

    __board_frames: list[str]
    __board_counts: dict[str, int]
    __board_from_str: dict[str, Board]

    filled: bool  # Whether memorizer finished accumulating frames

    def __init__(self, buffer_size: int):
        self.__memorize_count = buffer_size
        self.__board_frames = []
        self.__board_counts = defaultdict(int)
        self.__board_from_str = dict()
        self.filled = False
        self.__append_board(Board.get_empty_board())

    def __append_board(self, board: Board):
        board_str = board.to_str()
        self.__board_frames.append(board_str)
        self.__board_counts[board_str] += 1
        self.__board_from_str[board_str] = board

    def __pop_last(self):
        board_str = self.__board_frames.pop(0)
        self.__board_counts[board_str] -= 1

    def get_max_board(self) -> Board:
        max_board_str = max(self.__board_counts, key=self.__board_counts.get)
        return self.__board_from_str[max_board_str]

    def update(self, board: Board):
        if len(self.__board_frames) <= self.__memorize_count:
            self.__append_board(board)
        else:
            self.filled = True
            self.__append_board(board)
            self.__pop_last()

    def clear(self):
        self.__board_frames.clear()
        self.__board_counts.clear()
        self.__board_from_str.clear()
        self.filled = False
        self.__append_board(Board.get_empty_board())
