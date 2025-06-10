from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot
from Elements import *
from Elements import Board
from config import GLOBAL_CONFIG
from extra.types import ImageNP
from extra import factories


class ReaderWorker(QObject):
    __reader: ShogiBoardReader

    # Signal that sends processed/predicted data:
    frame_processed = pyqtSignal(
        ImageNP,  # Full image
        ImageNP,  # No perspective image
        Board,  # Predicted board object
        str,  # kif string
        BoardChangeStatus  # board update status
    )

    def __init__(self):
        super().__init__()
        self.__reader = factories.image_reader()

    @pyqtSlot()
    def send_data(self):
        if GLOBAL_CONFIG.Settings.predict_board:
            self.__reader.update()

        full_img = self.__reader.get_full_img(
            show_borders=GLOBAL_CONFIG.Visuals.show_borders,
            show_grid=GLOBAL_CONFIG.Visuals.show_grid,
            show_inventories=GLOBAL_CONFIG.Visuals.show_inventories
        )
        no_perspective = self.__reader.get_board_image_no_perspective(show_grid=True)
        predicted_board = self.__reader.get_board()
        kif = self.__reader.get_kif()
        update_status = self.__reader.get_last_update_status()
        self.frame_processed.emit(
            full_img.copy(),
            no_perspective.copy(),
            predicted_board,
            kif,
            update_status
        )

    def set_reader(self, reader: ShogiBoardReader) -> None:
        self.__reader = reader
        self.send_data()

    def get_reader(self) -> ShogiBoardReader:
        return self.__reader