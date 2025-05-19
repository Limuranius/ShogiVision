from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import QWidget, QListWidgetItem
import pypdfium2

from Elements import ShogiBoardReader
from Elements.Board import Board
from Elements.ImageGetters import Photo
from GUI.UI.UI_ScanBook import Ui_scan_book
from GUI.widgets.BookPageRangeDialog import BookPageRangeDialog
from extra import pdf_scan, factories
from GUI.widgets.ProgressBarDialog import ProgressBarDialog
from extra.types import Box


class ScanBook(QWidget):
    # page number, boards on this page, boxes around boards on this page
    __data: list[tuple[int, list[Board], list[Box]]]

    # path to current pdf
    __pdf_path: str

    # current index
    __data_i: int

    __reader: ShogiBoardReader

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_scan_book()
        self.ui.setupUi(self)
        self.setAcceptDrops(True)
        self.__data = []
        self.__data_i = 0
        self.update_pagination()
        self.__reader = factories.book_reader()

    @pyqtSlot()
    def on_previous_clicked(self):
        """Previous page button clicked"""
        self.__data_i -= 1
        self.update_pagination()
        self.update_pdf_page()

    @pyqtSlot()
    def on_next_clicked(self):
        """Next page button clicked"""
        self.__data_i += 1
        self.update_pagination()
        self.update_pdf_page()

    @pyqtSlot(int)
    def on_page_number_changed(self, page_number: int):
        min_page = self.__data[0][0]
        self.__data_i = page_number - min_page
        self.update_pagination()
        if len(self.__data) != 0:
            self.update_pdf_page()

    @pyqtSlot(QtGui.QDragEnterEvent)
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    @pyqtSlot(QtGui.QDropEvent)
    def dropEvent(self, event: QtGui.QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        pdf_path = files[0]

        win = BookPageRangeDialog(pdf_path)
        win.setModal(True)
        win.book_accepted.connect(self.load_pdf)
        win.exec()

    def update_pdf_page(self):
        pdf = pypdfium2.PdfDocument(self.__pdf_path)
        page, boards, boxes = self.__data[self.__data_i]
        img = pdf_scan.get_pdf_page_image(pdf, page - 1)

        self.ui.label_pdf_image.set_image(img)
        callbacks = []
        for board in boards:
            def set_board(board=board):
                self.ui.board_view.set_board(board)
            callbacks.append(set_board)
        self.ui.label_pdf_image.set_clickable_boxes(list(zip(boxes, callbacks)))

    def update_pagination(self):
        self.ui.pushButton_next.setDisabled(False)
        self.ui.pushButton_previous.setDisabled(False)
        if len(self.__data) == 0:
            self.ui.pushButton_next.setDisabled(True)
            self.ui.pushButton_previous.setDisabled(True)
        else:
            page_number = self.__data[self.__data_i][0]
            self.ui.spinBox_page_number.blockSignals(True)
            self.ui.spinBox_page_number.setValue(page_number)
            self.ui.spinBox_page_number.blockSignals(False)
            if self.__data_i == 0:
                self.ui.pushButton_previous.setDisabled(True)
            if self.__data_i == len(self.__data) - 1:
                self.ui.pushButton_next.setDisabled(True)

    def load_pdf(self, pdf_path: str, page_from: int, page_to: int):
        self.__data.clear()

        page_from -= 1
        page_to -= 1
        self.__pdf_path = pdf_path
        pdf = pypdfium2.PdfDocument(pdf_path)
        total = page_to - page_from + 1

        def pbar_func():
            data = []
            for page_number in range(page_from, page_to + 1):
                page_img = pdf_scan.get_pdf_page_image(pdf, page_number)
                boards_imgs = pdf_scan.extract_boards_images(page_img)
                boards_boxes = pdf_scan.extract_boards_boxes(page_img)
                boards = []
                for board_img in boards_imgs:
                    self.__reader.get_board_splitter().set_image_getter(Photo(board_img))
                    self.__reader.update()
                    board = self.__reader.get_board()
                    boards.append(board)
                data.append((page_number + 1, boards, boards_boxes))
                yield
            yield data

        def on_finish(result_data):
            self.__data = result_data
            min_page = self.__data[0][0]
            max_page = self.__data[-1][0]
            self.ui.spinBox_page_number.setMinimum(min_page)
            self.ui.spinBox_page_number.setMaximum(max_page)
            self.update_pdf_page()
            self.update_pagination()

        pbar = ProgressBarDialog(
            total=total,
            worker_func=pbar_func()
        )
        pbar.work_done.connect(on_finish)
        pbar.setModal(True)
        pbar.exec()
        pdf.close()



