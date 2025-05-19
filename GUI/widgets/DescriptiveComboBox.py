from PyQt5.QtCore import pyqtSignal, pyqtSlot, QVariant
from PyQt5.QtWidgets import QWidget
from GUI.UI.UI_DescriptiveComboBox import Ui_descriptiveComboBox
import copy


class DescriptiveComboBox(QWidget):
    """
    Combobox with title and description for each element in combobox
    """

    # Signal. Emits copy of newly selected object from __values
    element_changed = pyqtSignal(QVariant)

    # Text, Description, Value
    __values: list[tuple[str, str, object]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_descriptiveComboBox()
        self.ui.setupUi(self)
        self.__values = []

    def set_values(self, values: list[tuple[str, str, object]]):
        """Sets values of combobox
        :values contains name of object, its description and value that will be emitted when this object is selected
        """
        self.__values = values
        for text, _, _ in values:
            self.ui.comboBox.addItem(text)

    def set_name(self, name: str):
        """Sets title of combobox"""
        self.ui.label_name.setText(name + ":")

    @pyqtSlot(int)
    def on_element_changed(self, index: int):
        text, description, value = self.__values[index]
        value_copy = copy.copy(value)
        self.ui.label_description.setText(description)
        self.element_changed.emit(QVariant(value_copy))

    def switch_to_same_class(self, obj: object, emit_signal: bool = True):
        """
        Sets current value in combobox with one that has same class as :obj
        If :emit_signal flag is set to False then this switch of objects won't emit signal
        """
        for i, (_, _, value) in enumerate(self.__values):
            if type(obj) is type(value):  # Have same class
                if emit_signal:
                    self.ui.comboBox.setCurrentIndex(i)
                else:
                    self.ui.comboBox.blockSignals(True)
                    self.ui.comboBox.setCurrentIndex(i)
                    self.ui.comboBox.blockSignals(False)
