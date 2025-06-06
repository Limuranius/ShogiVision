from extra.types import FigureBoard, DirectionBoard
from .Move import *
from ..Board import Board, BoardChangeStatus
from .BoardsTree import BoardsTree


class BoardMemorizerTree:
    _tree: BoardsTree

    update_status: BoardChangeStatus = BoardChangeStatus.VALID_MOVE

    _patience: int
    _patience_stack: list[Board]

    def __init__(self, patience=100):
        self._tree = BoardsTree()

        # Inserting two sides of board since we don't know who goes first
        default_board_up = Board.default_board()
        default_board_down = Board.default_board()
        default_board_down.turn = Direction.DOWN
        self._tree.nodes_by_turn = [default_board_up, default_board_down]

        self._patience = patience
        self._patience_stack = []

    def update(self, figures: FigureBoard, directions: DirectionBoard, certainty_score: float = 1.0) -> None:
        """Updates board and stores status of update in 'update_status' variable"""
        new_board = Board(figures, directions)
        if certainty_score < 0.99:
            self.update_status = BoardChangeStatus.LOW_CERTAINTY
            return

        status = self._tree.insert(new_board)
        self.update_status = status

        if status in [BoardChangeStatus.VALID_MOVE, BoardChangeStatus.NOTHING_CHANGED]:
            self._patience_stack = [new_board]
        else:
            self._patience_stack.append(new_board)
            if len(self._patience_stack) > self._patience:
                self.update_status = BoardChangeStatus.NEED_MANUAL
                self._fill_missing_boards()


    def get_board(self) -> Board:
        return self._tree.get_last_boards()[0]

    def get_kif(self) -> str:
        s = """手合割：平手
先手：
後手：
手数----指手----消費時間--
"""
        flip = self.get_moves()[0].direction == Direction.DOWN
        for i, move in enumerate(self.get_moves()):
            signature = move.to_kif(flip=flip)
            row_fmt = "{:>4} {}\n"
            s += row_fmt.format(i + 1, signature)

        return s

    def clear(self):
        self._tree.clear()

    def get_moves(self) -> list[Move]:
        return self._tree.move_histories()[0]

    def _fill_missing_boards(self) -> None:
        """Bruteforces boards until they match with :target_board"""
        start_board = self.get_board()
        for i, target_board in enumerate(self._patience_stack):
            missing_boards = start_board.fill_missing_boards(target_board)
            if missing_boards is None:
                continue
            else:  # found missing boards
                if len(missing_boards) == 1:
                    missing_boards = missing_boards[0]
                else:
                    for path_i, path in enumerate(missing_boards):
                        path = [start_board] + path + [target_board]
                        moves = []
                        for ii in range(1, len(path)):
                            path[ii - 1].show_difference(path[ii])
                            moves.append(path[ii-1].get_move(path[ii]).to_usi())
                        print(f"[{path_i}] {' '.join(moves)}")
                    input_path_i = int(input("Enter path index: "))
                    missing_boards = missing_boards[input_path_i]

                tmp = self._patience_stack.copy()
                tmp = tmp[i:]
                for b in missing_boards + tmp:  # Trying to update with newly found boards and old boards from patience stack
                    self.update(b.figures, b.directions)
                return
        # raise Exception("Could not find missing boards")

    def provide_manual_info(self, moves: list[Move]) -> None:
        board = self.get_board()

        tmp = self._patience_stack.copy()

        for move in moves:
            board = board.make_move(move)
            self.update(board.figures, board.directions)
            print(self.update_status)
        for b in tmp:  # Trying to update with newly found boards and old boards from patience stack
            self.update(b.figures, b.directions)


