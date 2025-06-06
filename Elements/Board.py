from __future__ import annotations

import collections
import copy
import itertools
from enum import Enum
from typing import Any, Generator

import cv2
import numpy as np

from Elements.BoardMemorizer import BoardRules
from Elements.BoardMemorizer.Move import Move
from extra import utils
from extra.figures import Figure, Direction, get_figure_image
from extra.types import FigureBoard, DirectionBoard, ImageNP, Inventory


class BoardChangeStatus(Enum):
    INVALID_MOVE = "Invalid change"
    ILLEGAL_MOVE = "Illegal move"
    VALID_MOVE = "Valid move"
    NOTHING_CHANGED = "Board did not change"
    ACCUMULATING_DATA = "Accumulating data"
    LOW_CERTAINTY = "Low certainty"
    NEED_MANUAL = "Needs manual information"


class Board:
    figures: FigureBoard
    directions: DirectionBoard
    inventory_black: Inventory | None
    inventory_white: Inventory | None
    inventories: dict[Direction, Inventory | None]
    turn: Direction  # Which side has the move

    def __init__(
            self,
            figures: FigureBoard,
            directions: DirectionBoard,
            inventory_black: Inventory = None,
            inventory_white: Inventory = None,
            turn: Direction = Direction.UP,
    ):
        self.figures = figures
        self.directions = directions
        self.inventory_black = inventory_black
        self.inventory_white = inventory_white
        self.inventories = {Direction.UP: inventory_black, Direction.DOWN: inventory_white}
        self.turn = turn

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

    @classmethod
    def get_empty_board(cls):
        figures = [[Figure.EMPTY] * 9 for _ in range(9)]
        directions = [[Direction.NONE] * 9 for _ in range(9)]
        return Board(figures, directions)

    @classmethod
    def default_board(cls):
        row_main = lambda: [Figure.LANCE, Figure.KNIGHT, Figure.SILVER, Figure.GOLD, Figure.KING, Figure.GOLD,
                            Figure.SILVER, Figure.KNIGHT, Figure.LANCE]
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
        return Board(
            figures,
            directions,
            inventory_black=collections.defaultdict(int),
            inventory_white=collections.defaultdict(int)
        )

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

    def to_sfen(self) -> str:
        to_char = {
            Figure.PAWN: "p",
            Figure.KING: "k",
            Figure.LANCE: "l",
            Figure.KNIGHT: "n",
            Figure.SILVER: "s",
            Figure.GOLD: "g",
            Figure.BISHOP: "b",
            Figure.ROOK: "r",
        }
        sfen = ""

        # Board to sfen
        for i in range(9):
            row_str = "ඞ"  # dummy first character to not write if for first empty cell
            for j in range(9):
                figure = self.figures[i][j]
                direction = self.directions[i][j]
                if figure == Figure.EMPTY:
                    if row_str[-1].isdigit():  # previous cell was empty too
                        row_str = row_str[:-1] + str(int(row_str[-1]) + 1)
                    else:
                        row_str += "1"
                else:
                    if figure.is_promoted():
                        fig_chr = "+" + to_char[figure.unpromoted()]
                    else:
                        fig_chr = to_char[figure]
                    if direction == Direction.UP:
                        fig_chr = fig_chr.upper()
                    else:
                        fig_chr = fig_chr.lower()
                    row_str += fig_chr
            sfen += row_str[1:] + "/"
        sfen = sfen[:-1]  # removing last "/"

        # Side to sfen
        sfen += " "
        if self.turn == Direction.UP:
            sfen += "b"
        else:
            sfen += "w"
        sfen += " "

        # Inventories to sfen
        if self.inventory_black:
            for figure in self.inventory_black:
                count = self.inventory_black[figure]
                if count > 0:
                    if count > 1:
                        sfen += str(count)
                    sfen += to_char[figure].upper()
        if self.inventory_white:
            for figure in self.inventory_white:
                count = self.inventory_white[figure]
                if count > 0:
                    if count > 1:
                        sfen += str(count)
                    sfen += to_char[figure].lower()

        sfen += " 1"
        return sfen

    def lishogi_link(self) -> str:
        sfen = self.to_shogi_board().sfen()
        sfen = sfen.replace(" ", "_")
        url = f"https://lishogi.org/editor/{sfen}"
        return url

    def __str__(self):
        return self.to_sfen()

    def copy(self):
        return Board(
            copy.deepcopy(self.figures),
            copy.deepcopy(self.directions),
            copy.deepcopy(self.inventory_black),
            copy.deepcopy(self.inventory_white),
            self.turn,
        )

    def __hash__(self):
        return hash(str(self))

    def cells_equal(self, other: Board) -> bool:
        for i in range(9):
            for j in range(9):
                if self.figures[i][j] == other.figures[i][j] == Figure.EMPTY:
                    continue
                if self.figures[i][j] != other.figures[i][j] or self.directions[i][j] != other.directions[i][j]:
                    return False
        return True

    def __eq__(self, other):
        # return hash(self) == hash(other)  # compares everything: board, inventory, turn
        return self.cells_equal(other)  # compares only cells, without inventory and turn

    def get_changed_cells(self, new_board: Board) -> list[tuple[int, int]]:
        """Returns list of cells that changed from old_board to new_board"""
        changed = []
        for i in range(9):
            for j in range(9):
                figures_match = self.figures[i][j] == new_board.figures[i][j]
                directions_match = self.directions[i][j] == new_board.directions[i][j]
                cells_empty = self.figures[i][j] == new_board.figures[i][j] == Figure.EMPTY
                if cells_empty:
                    continue
                elif not figures_match or not directions_match:
                    changed.append((i, j))
        return changed

    def get_move(self, new_board: Board) -> Move | None:
        """
        Returns move taken between current board and new board
        or None if change from old_board to new_board is impossible in 1 move
        """
        changed_cells = self.get_changed_cells(new_board)
        changed_count = len(changed_cells)

        if changed_count == 1:
            """
            Valid options:
                piece dropped (EMPTY -> PIECE)
            """
            i, j = changed_cells[0]
            dropped_fig = new_board.figures[i][j]
            if (self.figures[i][j] == Figure.EMPTY
                    and dropped_fig not in [Figure.EMPTY, Figure.KING]
                    and not dropped_fig.is_promoted()):
                return Move(
                    array_destination=(i, j),
                    is_drop=True,
                    figure=dropped_fig,
                    direction=new_board.directions[i][j]
                )
            else:
                return None
        elif changed_count == 2:
            """
            Valid options: 
                piece moved (PIECE-EMPTY -> EMPTY-PIECE or EMPTY-PIECE -> PIECE-EMPTY), 
                piece took piece (PIECE1-PIECE2 -> EMPTY-PIECE1 or PIECE1-PIECE2 -> PIECE2-EMPTY),
                piece moved and promoted (PIECE-EMPTY -> EMPTY-PIECE_PROM or EMPTY-PIECE -> PIECE_PROM-EMPTY)
                piece took and promoted (PIECE1-PIECE2 -> EMPTY-PIECE1_PROM or PIECE1-PIECE2 -> PIECE2_PROM-EMPTY)
            """
            i1, j1 = changed_cells[0]
            i2, j2 = changed_cells[1]

            figure_1_old = self.figures[i1][j1]
            figure_2_old = self.figures[i2][j2]
            figure_1_new = new_board.figures[i1][j1]
            figure_2_new = new_board.figures[i2][j2]
            direction_1_old = self.directions[i1][j1] if figure_1_old != Figure.EMPTY else Direction.NONE
            direction_2_old = self.directions[i2][j2] if figure_2_old != Figure.EMPTY else Direction.NONE
            direction_1_new = new_board.directions[i1][j1] if figure_1_new != Figure.EMPTY else Direction.NONE
            direction_2_new = new_board.directions[i2][j2] if figure_2_new != Figure.EMPTY else Direction.NONE

            is_promotion = False

            # First piece moved
            # PIECE-EMPTY -> EMPTY-PIECE
            # PIECE1-PIECE2 -> EMPTY-PIECE1
            if figure_1_new == Figure.EMPTY and figure_2_new != Figure.EMPTY and figure_2_new == figure_1_old:
                if direction_1_old != direction_2_new:
                    return None
                moved_figure = figure_2_new
                moved_direction = direction_2_new
                x_origin = j1
                y_origin = i1
                x_destination = j2
                y_destination = i2

            # Second piece moved
            # EMPTY-PIECE -> PIECE-EMPTY
            # PIECE1-PIECE2 -> PIECE2-EMPTY
            elif figure_2_new == Figure.EMPTY and figure_1_new != Figure.EMPTY and figure_1_new == figure_2_old:
                if direction_2_old != direction_1_new:
                    return None
                moved_figure = figure_1_new
                moved_direction = direction_1_new
                x_origin = j2
                y_origin = i2
                x_destination = j1
                y_destination = i1

            # First piece moved with promotion
            # PIECE-EMPTY -> EMPTY-PIECE_PROM (PIECE != Gold / King)
            # PIECE1-PIECE2 -> EMPTY-PIECE1_PROM (PIECE1 != Gold / King)
            elif (
                    figure_1_new == Figure.EMPTY
                    and figure_2_new != Figure.EMPTY
                    and figure_1_old.is_promotable()
                    and figure_2_new == figure_1_old.promoted()
            ):
                if direction_1_old != direction_2_new:
                    return None
                moved_figure = figure_1_old
                moved_direction = direction_1_old
                x_origin = j1
                y_origin = i1
                x_destination = j2
                y_destination = i2
                is_promotion = True

            # Second piece moved with promotion
            # EMPTY - PIECE -> PIECE_PROM - EMPTY (PIECE != Gold / King)
            # PIECE1 - PIECE2 -> PIECE2_PROM - EMPTY (PIECE2 != Gold / King)
            elif (
                    figure_2_new == Figure.EMPTY
                    and figure_1_new != Figure.EMPTY
                    and figure_2_old.is_promotable()
                    and figure_1_new == figure_2_old.promoted()
            ):
                if direction_2_old != direction_1_new:
                    return None
                moved_figure = figure_2_old
                moved_direction = direction_2_old
                x_origin = j2
                y_origin = i2
                x_destination = j1
                y_destination = i1
                is_promotion = True

            else:
                return None

            return Move(
                array_destination=(y_destination, x_destination),
                array_origin=(y_origin, x_origin),
                is_promotion=is_promotion,
                figure=moved_figure,
                direction=moved_direction,
            )

        else:
            return None

    def get_change_status(self, new_board: Board) -> BoardChangeStatus:
        """Calculates difference between current and new board and returns comparison status"""
        if new_board == self:
            return BoardChangeStatus.NOTHING_CHANGED
        move = self.get_move(new_board)
        if move is None:
            return BoardChangeStatus.INVALID_MOVE
        if BoardRules.check_move(self, move):
            return BoardChangeStatus.VALID_MOVE
        return BoardChangeStatus.ILLEGAL_MOVE

    def is_sequence_valid(self, boards: list[Board], allow_repeats=True) -> bool:
        """
        From current state of board checks if sequence of boards is a valid continuation
        This means that if any board produces invalid or illegal move then sequence is considered invalid
        :allow_repeats - if True then NOTHING_CHANGED status is considered valid
        """
        bad_statuses = [BoardChangeStatus.INVALID_MOVE, BoardChangeStatus.ILLEGAL_MOVE]
        if not allow_repeats:
            bad_statuses.append(BoardChangeStatus.NOTHING_CHANGED)
        curr_board = self
        for next_board in boards:
            if curr_board.get_change_status(next_board) in bad_statuses:
                return False
            curr_board = next_board
        return True

    def get_cell_moves(self, i: int, j: int) -> list[Move]:
        """
        Returns all possible moves for cell (i, j)
        If cell is empty returns all drops on this cell
        If cell is enemy's piece returns empty list
        If cell is current player's piece returns all valid moves of this figure
        """
        moves = []
        if self.figures[i][j] == Figure.EMPTY:
            # Cell empty. Checking what figures we can drop here
            inv = self.inventories[self.turn]
            if inv is None:
                figures_to_drop = [fig for fig in Figure if fig.is_droppable()]
            else:
                figures_to_drop = [fig for fig in inv if inv[fig] > 0]
            for figure in figures_to_drop:
                moves.append(Move(
                    array_destination=(i, j),
                    figure=figure,
                    direction=self.turn,
                    is_drop=True,
                ))
        else:
            if self.directions[i][j] != self.turn:  # Enemy's figure
                return []
            figure = self.figures[i][j]
            for dx, dy in figure.get_moves(self.turn):
                new_i, new_j = i + dy, j + dx
                if 0 <= new_i < 9 and 0 <= new_j < 9:  # in bounds
                    not_friendly_fire = any([
                        self.figures[i][j] == Figure.EMPTY,
                        self.directions[new_i][new_j] == self.directions[i][j].opposite(),
                    ])
                    if not_friendly_fire:
                        moves.append(Move(
                            array_destination=(new_i, new_j),
                            figure=figure,
                            direction=self.turn,
                            array_origin=(i, j),
                            is_promotion=False
                        ))

                        # Also add move with promotion when possible
                        if self.turn == Direction.UP:
                            can_promote = i < 3 or new_i < 3  # on rows 0, 1, 2
                        else:
                            can_promote = i > 5 or new_i > 5  # on rows 6, 7, 8
                        if can_promote and figure.is_promotable():
                            moves.append(Move(
                                array_destination=(new_i, new_j),
                                figure=figure,
                                direction=self.turn,
                                array_origin=(i, j),
                                is_promotion=True
                            ))
        return moves


    def next_boards(self, subset: list[tuple[int, int]] = None) -> Generator[Board, Any, None]:
        """
        Yields all possible boards after 1 move
        :subset specifies certain cells to be checked to reduce number of moves
        """
        if subset is None:
            y = range(9)
            x = range(9)
        else:
            y = [i[0] for i in subset]
            x = [i[1] for i in subset]
        for i in y:
            for j in x:
                moves = self.get_cell_moves(i, j)
                for move in moves:
                    new_board = self.make_move(move)
                    yield new_board


    def continue_board_history(self, other_board: Board) -> Board:
        pass

    def make_move(self, move: Move) -> Board:
        """Makes move and returns new version of board"""
        new_board = self.copy()
        j, i = move.array_destination
        new_board.directions[i][j] = move.direction
        assert self.turn == move.direction  # is right side to move?
        if move.is_drop:
            new_board.figures[i][j] = move.figure
            if self.inventories[self.turn] is not None:  # Check if figure in inventory
                assert self.inventories[self.turn][move.figure] > 0
                self.inventories[self.turn][move.figure] -= 1
        else:
            if self.figures[i][j] != Figure.EMPTY:  # If take
                assert self.directions[i][j] != self.turn  # Check not friendly fire
                if self.inventories[self.turn] is not None:
                    self.inventories[self.turn][self.figures[i][j]] += 1  # Add taken figure to inventory
            if move.is_promotion:
                new_board.figures[i][j] = move.figure.promoted()
            else:
                new_board.figures[i][j] = move.figure
            orig_j, orig_i = move.array_origin
            new_board.figures[orig_i][orig_j] = Figure.EMPTY
        new_board.turn = self.turn.opposite()
        return new_board

    def fill_missing_boards(self, target_board: Board) -> list[list[Board]] | None:
        if len(self.get_changed_cells(target_board)) > 4:
            # Too many cells changed. Either board was obstructed or too many moves were performed
            return None
        diff = self.get_changed_cells(target_board)

        import tqdm
        # self.show_difference(target_board)

        pbar = tqdm.tqdm(desc="Filling missing boards")

        results = []

        # Trying depth 1
        for inner_board in self.next_boards(subset=diff):
            pbar.update(1)
            if self.is_sequence_valid([inner_board, target_board]):
                results.append([inner_board])

        # Trying depth 2
        for inner_board_1 in self.next_boards(subset=diff):
            for inner_board_2 in inner_board_1.next_boards(subset=diff):
                pbar.update(1)
                if self.is_sequence_valid([inner_board_1, inner_board_2, target_board]):
                    results.append([inner_board_1, inner_board_2])

        if len(results) == 0:
            pbar.set_description(pbar.desc + " Failed!")
            pbar.close()
            return None
        else:
            pbar.set_description(pbar.desc + f" Found {len(results)} possible paths")
            pbar.close()
            return results

    def show_difference(self, other: Board):
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(ncols=2)
        ax[0].imshow(self.to_image())
        ax[1].imshow(other.to_image())
        diff = self.get_changed_cells(other)
        plt.title(str(diff))
        plt.show()