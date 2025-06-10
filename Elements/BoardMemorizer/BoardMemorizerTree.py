from extra.types import FigureBoard, DirectionBoard
from Elements.Board.Move import *
from ..Board.Board import Board, BoardChangeStatus
from .BoardsTree import BoardsTree, BoardNode


class BoardMemorizerTree:
    _tree: BoardsTree

    update_status: BoardChangeStatus = BoardChangeStatus.VALID_MOVE

    _patience: int
    _patience_stack: list[Board]


    def __init__(self, patience=None):
        self._tree = BoardsTree()

        # Inserting two sides of board since we don't know who goes first
        default_board_up = Board.default_board()
        default_board_down = Board.default_board()
        default_board_down.turn = Direction.DOWN
        self._tree.nodes_by_turn = [[BoardNode(default_board_up), BoardNode(default_board_down)]]

        self._patience = patience
        self._patience_stack = []

    def update(self, figures: FigureBoard, directions: DirectionBoard, certainty_score: float = 1.0, show=False) -> None:
        """Updates board and stores status of update in 'update_status' variable"""
        new_board = Board(figures, directions)

        if certainty_score < 0.99:
            self.update_status = BoardChangeStatus.LOW_CERTAINTY
            return

        status = self._tree.insert(new_board)
        self.update_status = status

        if show:
            import matplotlib.pyplot as plt
            plt.imshow(new_board.to_image())
            print(status)
            plt.show()

        if self._patience is not None:
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
        if self._patience is not None:
            for _ in range(10):  # Adding last boards
                if len(self._patience_stack) > 0:
                    self._fill_missing_boards()
                else:
                    break
        s = """手合割：平手
先手：
後手：
手数----指手----消費時間--
"""
        moves = self.get_moves()
        for i, move in enumerate(self.get_moves()):
            signature = move.to_kif(flip=moves[0].direction == Direction.DOWN)
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
                            moves.append(path[ii-1].get_move(path[ii]))
                        start_board.show_moves(moves)
                        print(f"[{path_i}] {' '.join([m.to_usi() for m in moves])}")
                    start_board.show_difference(target_board)
                    input_path_i = int(input("Enter path index: "))
                    missing_boards = missing_boards[input_path_i]

                tmp = self._patience_stack.copy()
                tmp = tmp[i:]
                for b in missing_boards + tmp:  # Trying to update with newly found boards and old boards from patience stack
                    self.update(b.figures, b.directions)
                return
        print("FAILED TO FIND MISSING BOARDS")
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


