import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog
from GUI.widgets.ImageLabel import ImageLabel
from extra.types import Figure, Direction
from extra.figures import get_figure_image
from PyQt5.QtCore import pyqtSignal


CELL_IMG_SIZE = 70


class CellSelectDialog(QDialog):
    """
    Dialog window with selection of all possible figures and directions
    """

    # Signal. Emits selected figure and direction
    choice_clicked = pyqtSignal(Figure, Direction)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup()

    def setup(self):
        self.grid = QtWidgets.QGridLayout(self)
        self.setLayout(self.grid)

        width = 4

        values = [
            [(figure, Direction.DOWN) for figure in Figure if figure != Figure.EMPTY],
            [(Figure.EMPTY, Direction.NONE)],
            [(figure, Direction.UP) for figure in Figure if figure != Figure.EMPTY],
        ]

        begin_row = 0
        for group in values:
            for i, (figure, direction) in enumerate(group):
                row = begin_row + i // width
                col = i % width
                img_label = ImageLabel(self)
                self.grid.addWidget(img_label, row, col)
                img = get_figure_image(figure, direction)
                img = cv2.resize(img, (CELL_IMG_SIZE, CELL_IMG_SIZE))
                img_label.set_image(img)

                def img_clicked_func(*args, figure=figure, direction=direction):
                    self.choice_clicked.emit(figure, direction)
                    self.close()
                img_label.clicked.connect(img_clicked_func)
            begin_row += len(group) // width + 1

