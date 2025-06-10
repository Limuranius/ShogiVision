from typing import Callable

import cv2
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication

from Elements import Board
from Elements.Board.BoardDrawer import BoardDrawer
from Elements.Board.Move import Move
from GUI.widgets.InteractiveImageLabel import InteractiveImageLabel
from extra import utils
from extra.figures import Figure, Direction, get_figure_image
from extra.types import ImageNP


class MovePicker(InteractiveImageLabel):
    """
    Widget used to pick move on board
    """

    move_picked = pyqtSignal(Move)  # emits iteration number
    do_moves: bool  # If moves are performed after being picked

    board: Board
    __last_click_info: dict
    __no_selection_image: ImageNP

    bboxes: dict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.board = Board.default_board()
        self.board.inventory_black[Figure.GOLD] = 3
        self.board.inventory_black[Figure.PAWN] = 5
        self.board.inventory_black[Figure.BISHOP] = 1
        self.board.inventory_white[Figure.LANCE] = 3
        self.board.inventory_white[Figure.KNIGHT] = 5
        self.board.inventory_white[Figure.ROOK] = 1

        self.set_board(self.board)

        self.__last_click_info = None
        self.do_moves = False

    def set_board(self, board: Board):
        self.board = board
        drawer = BoardDrawer(board)
        img = drawer.to_image()
        self.__no_selection_image = img
        self.set_image(img)

        # Creating bounding boxes and their callbacks
        bboxes = drawer.get_bounding_boxes()
        self.bboxes = bboxes
        boxes = []
        for i in range(9):
            for j in range(9):
                bbox = bboxes["board"][i][j]

                def cell_click(i=i, j=j):
                    click_info = {
                        "coords": (i, j),
                        "is_inventory": False,
                        "figure": board.figures[i][j],
                        "direction": board.directions[i][j]
                    }
                    self.__check_new_click(click_info)

                boxes.append((bbox, cell_click))
        for figure, bbox in bboxes["inventory_black"].items():
            def f(figure=figure):
                click_info = {
                    "coords": None,
                    "is_inventory": True,
                    "figure": figure,
                    "direction": Direction.UP
                }
                self.__check_new_click(click_info)

            boxes.append((bbox, f))
        for figure, bbox in bboxes["inventory_white"].items():
            def f(figure=figure):
                click_info = {
                    "coords": None,
                    "is_inventory": True,
                    "figure": figure,
                    "direction": Direction.DOWN
                }
                self.__check_new_click(click_info)

            boxes.append((bbox, f))
        self.set_clickable_boxes(boxes, outline_boxes=False)

    def __check_new_click(self, click_info: dict) -> None:
        self.clear_selection()

        if self.__last_click_info is None:
            # First ever click
            self.__last_click_info = click_info

        if click_info["is_inventory"]:
            self.__check_inventory_click(click_info)
        else:
            self.__check_board_click(click_info)

        self.__last_click_info = click_info


    def __check_inventory_click(self, click_info: dict):
        # Showing cells to move at
        free_cells = [(i, j) for i in range(9) for j in range(9) if self.board.figures[i][j] == Figure.EMPTY]
        if click_info["direction"] == self.board.turn:
            self.select_cells(free_cells)


    def __check_board_click(self, click_info: dict):
        self.show_cell_moves(*click_info["coords"])  # Showing cells to move at

        clicked_empty = click_info["figure"] == Figure.EMPTY
        prev_clicked_empty = self.__last_click_info["figure"] == Figure.EMPTY
        prev_inventory = self.__last_click_info["is_inventory"]
        prev_board = not prev_inventory
        prev_right_side = self.__last_click_info["direction"] == self.board.turn
        friendly_fire = click_info["direction"] == self.__last_click_info["direction"]

        if prev_inventory:
            if clicked_empty and prev_right_side:
                # Drop on empty cell
                self.emit_move(Move(
                    array_destination=click_info["coords"],
                    figure=self.__last_click_info["figure"],
                    direction=self.__last_click_info["direction"],
                    is_drop=True,
                ))
                self.__last_click_info = click_info
            return

        can_be_promotion = (
                prev_board
                and self.__last_click_info["figure"] != Figure.EMPTY
                and (
                        click_info["coords"][0] in self.__last_click_info["direction"].promotion_zone_array_y()
                        or self.__last_click_info["coords"][0] in self.__last_click_info[
                            "direction"].promotion_zone_array_y()
                )
                and self.__last_click_info["figure"].is_promotable()
        )
        dest_cells = [m.array_destination for m in self.board.get_cell_moves(*self.__last_click_info["coords"])]
        cell_in_reach = not prev_inventory and click_info["coords"] in dest_cells

        if prev_board and (not friendly_fire or clicked_empty) and cell_in_reach and not prev_clicked_empty:
            # Move on new cell
            if can_be_promotion:
                # Showing popup window with selection whether to promote or not
                move = Move(
                    array_destination=click_info["coords"],
                    figure=self.__last_click_info["figure"],
                    direction=self.__last_click_info["direction"],
                    array_origin=self.__last_click_info["coords"],
                    is_drop=False,
                )
                def prom_callback(is_promotion: bool, move=move):
                    move.is_promotion = is_promotion
                    self.emit_move(move)
                self.__promotion_selection_popup(
                    click_info["coords"][0],
                    click_info["coords"][1],
                    self.__last_click_info["figure"],
                    self.__last_click_info["direction"],
                    prom_callback,
                )
            else:
                self.emit_move(Move(
                    array_destination=click_info["coords"],
                    figure=self.__last_click_info["figure"],
                    direction=self.__last_click_info["direction"],
                    array_origin=self.__last_click_info["coords"],
                    is_drop=False,
                ))

    def show_cell_moves(self, i: int, j: int):
        moves = self.board.get_cell_moves(i, j)
        cells = [m.array_destination for m in moves]
        self.select_cells(cells)

    def select_cells(self, coords: list[tuple[int, int]]):
        img = self.__no_selection_image.copy()
        for (i, j) in coords:
            bbox = self.bboxes["board"][i][j]
            self.add_selection(img, *bbox)
        self.set_image(img)

    def clear_selection(self) -> None:
        # self.set_image(self.__no_selection_image)
        self.set_board(self.board)

    def __promotion_selection_popup(
            self,
            i: int, j: int,
            figure: Figure, direction: Direction,
            callback: Callable[[bool], None],  # callback with boolean of promotion
    ):
        img = self.__no_selection_image.copy()

        x, y, w, h = self.bboxes["board"][i][j]
        fig_img = get_figure_image(figure, direction)
        fig_prom_img = get_figure_image(figure.promoted(), direction)
        fig_img = cv2.resize(fig_img, (w // 2, h // 2))
        fig_prom_img = cv2.resize(fig_prom_img, (w // 2, h // 2))

        # left - unpromoted, right - promoted
        utils.overlay_image_on_image(img, fig_img, x, y)
        utils.overlay_image_on_image(img, fig_prom_img, x + w // 2, y)
        self.set_image(img)

        self.add_clickable_box((x, y, w // 2, h // 2), lambda: callback(False))
        self.add_clickable_box((x + w // 2, y, w // 2, h // 2), lambda: callback(True))

    def emit_move(self, move: Move) -> None:
        self.move_picked.emit(move)
        if self.do_moves:
            self.board = self.board.make_move(move)
            self.set_board(self.board)


if __name__ == '__main__':
    apps = QApplication([])
    b = MovePicker()
    b.do_moves = True
    b.show()
    apps.exec()
