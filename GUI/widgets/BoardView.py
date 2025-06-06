import webbrowser
import pyperclip
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget

from Elements.Board import Board
from GUI.UI.UI_BoardView import Ui_boardView


class BoardView(QWidget):
    """
    Widget that contains image of board stored in :__board
    and buttons to copy SFEN or open this board in lishogi
    """

    __board: Board

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_boardView()
        self.ui.setupUi(self)
        self.set_board(Board.get_empty_board())

    @pyqtSlot()
    def on_to_lishogi_clicked(self):
        """'To lishogi' button clicked"""
        sfen = self.__board.to_sfen()
        sfen = sfen.replace(" ", "_")
        url = f"https://lishogi.org/editor/{sfen}"
        webbrowser.open(url)

    @pyqtSlot()
    def on_copy_sfen_clicked(self):
        """'Copy sfen' button clicked"""
        sfen = self.__board.to_sfen()
        pyperclip.copy(sfen)

    def __show_board(self):
        board_image = self.__board.to_image()
        self.ui.board_image_label.set_image(board_image)

    def set_board(self, board: Board):
        """Sets :board to show in this widget"""
        self.__board = board
        self.__show_board()
