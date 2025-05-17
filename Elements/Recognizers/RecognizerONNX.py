import numpy as np
import onnxruntime

from ShogiNeuralNetwork import preprocessing
from ShogiNeuralNetwork.data_info import CATEGORIES_FIGURE_TYPE, CATEGORIES_DIRECTION
from extra.figures import Figure, Direction
from extra.image_modes import ImageMode
from extra.types import CellsImages, FigureBoard, DirectionBoard, ImageNP
from .Recognizer import Recognizer


class RecognizerONNX(Recognizer):
    model: onnxruntime.InferenceSession
    cell_img_size: int
    image_mode: ImageMode

    def __init__(self, model_path: str):
        self.model = onnxruntime.InferenceSession(model_path)

        shape = self.model.get_inputs()[0].shape
        self.cell_img_size = shape[1]
        if shape[-1] == 1:
            self.image_mode = ImageMode.GRAYSCALE
        else:
            self.image_mode = ImageMode.ORIGINAL

    def recognize_cell(self, cell_img: ImageNP) -> tuple[Figure, Direction]:
        inp = preprocessing.prepare_cell_img(
            cell_img,
            self.image_mode,
            self.cell_img_size
        )
        predictions = self.model.run(["figure", "direction"], {"input": inp})
        figure_label = predictions[0].argmax()
        direction_label = predictions[1].argmax()
        figure = CATEGORIES_FIGURE_TYPE[figure_label]
        direction = CATEGORIES_DIRECTION[direction_label]
        return figure, direction

    def recognize_board(self, cells_imgs: CellsImages) -> tuple[FigureBoard, DirectionBoard, float]:
        inp = preprocessing.prepare_cells_imgs(
            cells_imgs,
            self.image_mode,
            self.cell_img_size,
        )
        predictions = self.model.run(["figure", "direction"], {"input": inp})

        figure_predict = predictions[0].argmax(axis=1)
        direction_predict = predictions[1].argmax(axis=1)

        figure_score = predictions[0].max(axis=1).mean()
        direction_score = predictions[1].max(axis=1).mean()
        score = (figure_score + direction_score) / 2

        figure_predict = np.reshape(figure_predict, (9, 9))
        direction_predict = np.reshape(direction_predict, (9, 9))

        figures = [[Figure.EMPTY for _ in range(9)] for __ in range(9)]
        directions = [[Direction.NONE for _ in range(9)] for __ in range(9)]

        for i in range(9):
            for j in range(9):
                figure_label = figure_predict[i][j]
                direction_label = direction_predict[i][j]
                figure = CATEGORIES_FIGURE_TYPE[figure_label]
                direction = CATEGORIES_DIRECTION[direction_label]
                figures[i][j] = figure
                directions[i][j] = direction
        return figures, directions, score
