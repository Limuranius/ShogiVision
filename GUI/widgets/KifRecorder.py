from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget
from GUI.UI.UI_KifRecorder import Ui_kifRecorder
import pyperclip


class KifRecorder(QWidget):
    """
    Widget with text field to record KIF
    and copy button to copy KIF
    """

    __kif: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_kifRecorder()
        self.ui.setupUi(self)
        self.__kif = ""

    def set_kif(self, kif: str):
        self.__kif = kif
        self.ui.textEdit_kif.setText(self.__kif)

    @pyqtSlot()
    def on_copy_kif_clicked(self):
        pyperclip.copy(self.__kif)
