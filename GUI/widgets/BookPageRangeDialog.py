import pypdfium2
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QDialog

from GUI.UI.UI_BookPageRangeDialog import Ui_Dialog


class BookPageRangeDialog(QDialog):
    """
    Dialog window that opens up when trying to load pdf
    Has fields to specify range of pages that need to be loaded
    """

    __pdf_path: str

    # Emitted when user accepts dialog window
    # Emits path to pdf, page from, page to
    book_accepted = pyqtSignal(str, int, int)

    def __init__(self, pdf_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.__pdf_path = pdf_path
        pdf = pypdfium2.PdfDocument(pdf_path)
        total_pages = len(pdf)
        self.ui.label_total_pages.setText(f"Total pages: {total_pages}")
        self.ui.spinBox_from.setMaximum(total_pages)
        self.ui.spinBox_to.setMaximum(total_pages)
        self.ui.spinBox_to.setValue(total_pages)

    @pyqtSlot(int)
    def from_changed(self, value: int):
        """When min page changed"""
        if value > self.ui.spinBox_to.value():
            self.ui.spinBox_to.setValue(value)

    @pyqtSlot(int)
    def to_changed(self, value: int):
        """When max page changed"""
        if value < self.ui.spinBox_from.value():
            self.ui.spinBox_from.setValue(value)

    @pyqtSlot()
    def accept(self):
        """OK button was clicked"""
        self.book_accepted.emit(
            self.__pdf_path,
            self.ui.spinBox_from.value(),
            self.ui.spinBox_to.value(),
        )
        self.close()
