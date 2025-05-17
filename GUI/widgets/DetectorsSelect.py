import copy

from PyQt5.QtCore import pyqtSlot, pyqtSignal, QVariant, QTimer
from PyQt5.QtWidgets import QWidget
from config.config import GLOBAL_CONFIG
import extra.utils
from Elements import HardcodedCornerDetector, BoardSplitter, CornerDetector
from GUI.UI.UI_DetectorsSelect import Ui_visualCornerSelect
from GUI.widgets import combobox_values
from extra import factories


class DetectorsSelect(QWidget):
    """
    Widget with board splitter configuration
    Contains:
        Original image taken from __splitter's image getter
        Cropped image of board with removed perspective (if __use_one_image set to False)
        Combobox with corner detectors
        Combobox with inventory detectors
        Ticks "Show borders", "Show grid", "Show inventories"
    """

    __record_corner_clicks: bool
    __splitter: BoardSplitter

    # if True then all information will be shown on one image
    __use_one_image: bool = False

    # Emits copy of new splitter when some parts of it change
    splitter_changed = pyqtSignal(QVariant)

    __label_disappear_timer: QTimer
    __update_images_timer: QTimer

    __clicked_corners: list[tuple[int, int]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_visualCornerSelect()
        self.ui.setupUi(self)
        self.__record_corner_clicks = False
        self.__splitter = factories.default_board_splitter()

        # Filling combo boxes
        cd_name, cd_values = combobox_values.corner_detector()
        self.ui.corner_detector_select.set_name(cd_name)
        self.ui.corner_detector_select.set_values(cd_values)
        id_name, id_values = combobox_values.inventory_detector()
        self.ui.inventory_detector_select.set_values(id_values)
        self.ui.inventory_detector_select.set_name(id_name)

        # Filling check boxes
        self.ui.checkBox_show_borders.setChecked(GLOBAL_CONFIG.Visuals.show_borders)
        self.ui.checkBox_show_grid.setChecked(GLOBAL_CONFIG.Visuals.show_grid)
        self.ui.checkBox_show_inventories.setChecked(GLOBAL_CONFIG.Visuals.show_inventories)

        self.ui.label_click_corners.setVisible(False)
        self.__label_disappear_timer = QTimer()
        self.__update_images_timer = QTimer()
        self.__update_images_timer.timeout.connect(self.update_images)

        self.__clicked_corners = []

    @pyqtSlot(QVariant)
    def on_corner_detector_changed(self, corner_detector: CornerDetector):
        if isinstance(corner_detector, HardcodedCornerDetector):
            self.ui.pushButton_set_corners.setVisible(True)
        else:
            self.ui.pushButton_set_corners.setVisible(False)
            self.__splitter.set_corner_detector(corner_detector)
            self.splitter_changed.emit(QVariant(copy.copy(self.__splitter)))
            self.update_images()

    @pyqtSlot()
    def on_set_corners_clicked(self):
        self.__clicked_corners.clear()
        self.__record_corner_clicks = True

        # Showing hint for several seconds
        self.ui.label_click_corners.setVisible(True)
        self.__label_disappear_timer.singleShot(5000, lambda: self.ui.label_click_corners.setVisible(False))

        if self.__use_one_image:
            self.ui.image_label_original.setVisible(True)
            self.ui.image_label_no_perspective.setVisible(False)

    @pyqtSlot(QVariant)
    def on_inventory_detector_changed(self, inventory_detector):
        self.__splitter.set_inventory_detector(inventory_detector)
        self.splitter_changed.emit(QVariant(copy.copy(self.__splitter)))

    @pyqtSlot(int, int, int, int)
    def on_image_clicked(self, view_x: int, view_y: int, orig_x: int, orig_y: int) -> None:
        if self.__record_corner_clicks:
            self.__clicked_corners.append((orig_x, orig_y))
            if len(self.__clicked_corners) == 4:
                corner_detector = HardcodedCornerDetector(tuple(self.__clicked_corners))
                self.__splitter.set_corner_detector(corner_detector)
                self.splitter_changed.emit(QVariant(copy.copy(self.__splitter)))
                self.__record_corner_clicks = False
                if self.__use_one_image:
                    self.ui.image_label_original.setVisible(False)
                    self.ui.image_label_no_perspective.setVisible(True)
            self.update_images()

    def update_images(self):
        no_persp = self.__splitter.get_board_image_no_perspective(draw_grid=GLOBAL_CONFIG.Visuals.show_grid)
        if self.__record_corner_clicks:
            full_img = self.__splitter.get_full_img()
            self.ui.image_label_original.set_image(
                extra.utils.draw_points(full_img, self.__clicked_corners)
            )
        else:
            full_img = self.__splitter.get_full_img(
                show_borders=GLOBAL_CONFIG.Visuals.show_borders,
                show_grid=GLOBAL_CONFIG.Visuals.show_grid,
                show_inventories=GLOBAL_CONFIG.Visuals.show_inventories
            )
            self.ui.image_label_original.set_image(full_img)
        self.ui.image_label_no_perspective.set_image(no_persp)

    def set_use_one_image(self, use_one_image: bool):
        self.__use_one_image = use_one_image
        self.ui.image_label_original.setVisible(False)

    def set_splitter(self, splitter: BoardSplitter):
        self.__splitter = splitter
        self.ui.corner_detector_select.switch_to_same_class(
            splitter.get_corner_detector(),
            emit_signal=False
        )
        self.ui.inventory_detector_select.switch_to_same_class(
            splitter.get_inventory_detector(),
            emit_signal=False
        )

    def start_continuous_update(self):
        self.__update_images_timer.start(100)

    def stop_continuous_update(self):
        self.__update_images_timer.stop()

    @pyqtSlot(bool)
    def on_show_borders_switched(self, show_borders: bool):
        GLOBAL_CONFIG.Visuals.show_borders = show_borders
        self.update_images()

    @pyqtSlot(bool)
    def on_show_grid_switched(self, show_grid: bool):
        GLOBAL_CONFIG.Visuals.show_grid = show_grid
        self.update_images()

    @pyqtSlot(bool)
    def on_show_inventories_switched(self, show_inventories: bool):
        GLOBAL_CONFIG.Visuals.show_inventories = show_inventories
        self.update_images()
