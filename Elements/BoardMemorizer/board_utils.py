import itertools
from typing import Any, Generator

from extra.figures import Direction
from .Move import *
from ..Board import Board


def get_changed_cells(old_board: Board, new_board: Board) -> list[tuple[int, int]]:
    """Returns list of cells that changed from old_board to new_board"""
    changed = []
    for i in range(9):
        for j in range(9):
            figures_match = old_board.figures[i][j] == new_board.figures[i][j]
            directions_match = old_board.directions[i][j] == new_board.directions[i][j]
            cells_empty = old_board.figures[i][j] == new_board.figures[i][j] == Figure.EMPTY
            if cells_empty:
                continue
            elif not figures_match or not directions_match:
                changed.append((i, j))
    return changed


def get_move(old_board: Board, new_board: Board) -> Move | None:
    """
    Returns move taken between old_board and new_board
    or None if change from old_board to new_board is impossible in 1 move
    """
    changed_cells = get_changed_cells(old_board, new_board)
    changed_count = len(changed_cells)

    if changed_count == 1:
        """
        Valid options:
            piece dropped (EMPTY -> PIECE)
        """
        i, j = changed_cells[0]
        dropped_fig = new_board.figures[i][j]
        if (old_board.figures[i][j] == Figure.EMPTY
                and dropped_fig not in [Figure.EMPTY, Figure.KING]
                and not dropped_fig.is_promoted()):
            x = j + 1
            y = i + 1
            return Move(
                destination=(x, y),
                is_drop=True,
                figure=dropped_fig
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

        figure_1_old = old_board.figures[i1][j1]
        figure_2_old = old_board.figures[i2][j2]
        figure_1_new = new_board.figures[i1][j1]
        figure_2_new = new_board.figures[i2][j2]
        direction_1_old = old_board.directions[i1][j1] if figure_1_old != Figure.EMPTY else Direction.NONE
        direction_2_old = old_board.directions[i2][j2] if figure_2_old != Figure.EMPTY else Direction.NONE
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
            x_origin = j1 + 1
            y_origin = i1 + 1
            x_destination = j2 + 1
            y_destination = i2 + 1

        # Second piece moved
        # EMPTY-PIECE -> PIECE-EMPTY
        # PIECE1-PIECE2 -> PIECE2-EMPTY
        elif figure_2_new == Figure.EMPTY and figure_1_new != Figure.EMPTY and figure_1_new == figure_2_old:
            if direction_2_old != direction_1_new:
                return None
            moved_figure = figure_1_new
            x_origin = j2 + 1
            y_origin = i2 + 1
            x_destination = j1 + 1
            y_destination = i1 + 1

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
            x_origin = j1 + 1
            y_origin = i1 + 1
            x_destination = j2 + 1
            y_destination = i2 + 1
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
            x_origin = j2 + 1
            y_origin = i2 + 1
            x_destination = j1 + 1
            y_destination = i1 + 1
            is_promotion = True

        else:
            return None

        return Move(
            destination=(x_destination, y_destination),
            origin=(x_origin, y_origin),
            is_promotion=is_promotion,
            figure=moved_figure
        )

    else:
        return None


def get_all_variants_bruteforce(board: Board) -> Generator[Board, Any, None]:
    """Brute forces all possible changes of 1-2 cells on board"""
    all_cell_variants = [
                            (f, d)
                            for f in set(Figure) - {Figure.EMPTY}
                            for d in [Direction.UP, Direction.DOWN]
                        ] + [(Figure.EMPTY, Direction.NONE)]
    for i1, j1, i2, j2, (f1, d1), (f2, d2) in itertools.product(
            range(9), range(9),
            range(9), range(9),
            all_cell_variants,
            all_cell_variants,
    ):
        new_board = board.copy()
        new_board.figures[i1][j1] = f1
        new_board.figures[i2][j2] = f2
        new_board.directions[i1][j1] = d1
        new_board.directions[i2][j2] = d2
        yield new_board


def get_all_variants_smart(board: Board) -> list[Board]:
    """Returns list of all possible boards after 1 move"""
    all_cell_variants = [
                            (f, d)
                            for f in set(Figure) - {Figure.EMPTY}
                            for d in [Direction.UP, Direction.DOWN]
                        ] + [(Figure.EMPTY, Direction.NONE)]
    king_moves = set([
        (dx, dy)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        if (dx, dy) != (0, 0)
    ])
    rook_moves = {
        *[(dx, 0) for dx in range(-8, 9) if dx != 0],
        *[(0, dy) for dy in range(-8, 9) if dy != 0]
    }
    bishop_moves = {
        *[(d, d) for d in range(-8, 9) if d != 0],
        *[(d, -d) for d in range(-8, 9) if d != 0],
    }
    knight_moves = {
        (dx, dy)
        for dx in range(-2, 3)
        for dy in range(-2, 3)
        if abs(dx) + abs(dy) == 3
    }
    figure_moves = {
        Figure.PAWN: rook_moves,
        Figure.KING: king_moves,
        Figure.LANCE: rook_moves,
        Figure.KNIGHT: knight_moves,
        Figure.SILVER: king_moves,
        Figure.GOLD: king_moves,
        Figure.BISHOP: bishop_moves,
        Figure.ROOK: rook_moves,
        Figure.PAWN_PROM: king_moves,
        Figure.LANCE_PROM: king_moves,
        Figure.KNIGHT_PROM: king_moves,
        Figure.SILVER_PROM: king_moves,
        Figure.BISHOP_PROM: king_moves | bishop_moves,
        Figure.ROOK_PROM: king_moves | rook_moves,
    }

    res = []
    for i in range(9):
        for j in range(9):
            if board.figures[i][j] == Figure.EMPTY:
                # bruteforcing drop on this cell
                for f, d in all_cell_variants:
                    new_board = board.copy()
                    new_board.figures[i][j] = f
                    new_board.directions[i][j] = d
                    res.append(new_board)
            else:
                # bruteforcing move from this cell
                f, d = board.figures[i][j], board.directions[i][j]
                for dx, dy in figure_moves[f]:
                    new_i, new_j = i + dy, j + dx
                    if 0 <= new_i < 9 and 0 <= new_j < 9:
                        new_board = board.copy()
                        new_board.figures[new_i][new_j] = f
                        new_board.directions[new_i][new_j] = d
                        new_board.figures[i][j] = Figure.EMPTY
                        new_board.directions[i][j] = Direction.NONE
                        res.append(new_board)
    return res
