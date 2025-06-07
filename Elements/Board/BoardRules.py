# For type annotations to avoid circular imports
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from Elements.Board import Board

from Elements.Board.Move import Move
from extra.figures import Figure, Direction


def check_move(board: Board, move: Move) -> bool:
    """Checks if :move is legal on :board"""
    # checking if move was made by correct side
    if board.turn is not None:
        valid_side = move.direction == board.turn
    else:
        valid_side = True

    if move.is_drop:
        # If move is drop
        drop_good = _check_drop(board, move)
        figure_move_good = True
    else:
        # If move is move
        drop_good = True
        figure_move_good = _check_figure_move(board, move)

    return all((
        valid_side,
        _has_moves_after(move),
        drop_good,
        figure_move_good,
    ))

def _check_drop(board: Board, move: Move) -> bool:
    """
    Check if cell was free
    Check if figure was in inventory
    Check if figure is droppable (not king, or promoted figures)
    """
    i, j = move.array_destination
    cell_free = board.figures[i][j] == Figure.EMPTY

    turn = move.direction
    inventory = board.inventories[turn]
    if inventory is None:
        has_in_inventory = True
    else:
        has_in_inventory = inventory[move.figure] > 0

    return all((
        cell_free,
        has_in_inventory,
        move.figure.is_droppable(),
    ))

def _has_moves_after(move: Move) -> bool:
    """Returns True if there is at least one cell that figure can go to after move"""
    figure = move.figure
    if move.is_promotion:
        figure = figure.promoted()
    all_moves = figure.get_moves(move.direction)
    all_pos = move.array_destination + all_moves
    pos_in_bounds = ((0 <= all_pos) & (all_pos <= 8)).all(axis=1)
    return pos_in_bounds.any()

def _check_figure_move(board: Board, move: Move) -> bool:
    """
    Check not friendly fire
    Check destination in list of figure moves
    Check promotion on enemy territory
    """
    dest_i, dest_j = move.array_destination
    orig_i, orig_j = move.array_origin

    # Not friendly fire
    is_take = board.figures[dest_i][dest_j] != Figure.EMPTY
    if not is_take:  # moved to free cell
        not_friendly_fire = True
    else:
        not_friendly_fire = board.directions[orig_i][orig_j] != board.directions[dest_i][dest_j]

    # Destination is reachable
    all_moves = move.figure.get_moves(move.direction)
    all_pos = move.array_origin + all_moves
    move_in_list = (np.abs(move.array_destination - all_pos).sum(axis=1) == 0).any()

    # Promotion on enemy territory
    if move.is_promotion:
        if move.direction == Direction.UP:
            promotion_valid = orig_i < 3 or dest_i < 3  # on rows 0, 1, 2
        else:
            promotion_valid = orig_i > 5 or dest_i > 5 # on rows 6, 7, 8
    else:
        promotion_valid = True

    return all((
        not_friendly_fire,
        move_in_list,
        promotion_valid,
    ))
