from enum import Enum, auto

import cv2
import numpy as np
import pyperclip
from PIL import ImageGrab
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, QVariant, pyqtSignal
from PyQt5.QtWidgets import QDialog, QFileDialog

from GUI.UI.UI_UploadFileDialog import Ui_Dialog
from GUI.widgets.error_mesage_box import error_message
from extra.types import ImageNP


class FileType(Enum):
    ONE_IMAGE = auto()
    MANY_IMAGES = auto()
    VIDEO = auto()


DRAG_ENTER_COLOR = "gray"


class UploadFileDialog(QDialog):
    __file_type: FileType

    """
    Is emitted after user uploaded files
    Output depends on file type:
        ONE_IMAGE - numpy.ndarray image
        MANY_IMAGES - list of numpy.ndarray images
        VIDEO - str, path to video
    """
    file_uploaded = pyqtSignal(QVariant)

    def __init__(self, file_type: FileType, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.__file_type = file_type
        self.setAcceptDrops(True)

        match file_type:
            case FileType.ONE_IMAGE:
                self.ui.label_file_type.setText("File type: one image")
            case FileType.MANY_IMAGES:
                self.ui.label_file_type.setText("File type: several images")
            case FileType.VIDEO:
                self.ui.label_file_type.setText("File type: video")

    @pyqtSlot()
    def on_paste_clicked(self):
        """Paste from clipboard button clicked"""
        match self.__file_type:
            case FileType.ONE_IMAGE:
                clipboard_data = ImageGrab.grabclipboard()
                if clipboard_data is None:  # On windows when no image in clipboard
                    error_message("Could not load image")
                    return
                elif isinstance(clipboard_data, list):  # On windows when copied files
                    img_path = clipboard_data[0]
                    success, image_np = self.__load_image_path(img_path)
                    if not success:
                        return
                else:  # When there is image in clipboard
                    image_np = np.array(clipboard_data)
                data = image_np
            case FileType.MANY_IMAGES:
                files_paths = pyperclip.paste().split("\n")
                success, data = self.__load_images_paths(files_paths)
                if not success:
                    return
            case FileType.VIDEO:
                video_path = pyperclip.paste()
                if not self.__is_video_path_valid(video_path):
                    return
                data = video_path
        self.__send_data(data)

    @pyqtSlot()
    def on_search_clicked(self):

        match self.__file_type:
            case FileType.ONE_IMAGE:
                img_path = QFileDialog.getOpenFileName(self)[0]
                success, data = self.__load_image_path(img_path)
                if not success:
                    return
            case FileType.MANY_IMAGES:
                files_paths = QFileDialog.getOpenFileNames(self)[0]
                success, data = self.__load_images_paths(files_paths)
                if not success:
                    return
            case FileType.VIDEO:
                video_path = QFileDialog.getOpenFileName(self)[0]
                if not self.__is_video_path_valid(video_path):
                    return
                data = video_path
        self.__send_data(data)

    @pyqtSlot(QtGui.QDragEnterEvent)
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet(f"background-color: {DRAG_ENTER_COLOR}")
        else:
            event.ignore()

    @pyqtSlot(QtGui.QDragLeaveEvent)
    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent):
        self.setStyleSheet("")

    @pyqtSlot(QtGui.QDropEvent)
    def dropEvent(self, event: QtGui.QDropEvent):
        files_paths = [u.toLocalFile() for u in event.mimeData().urls()]
        self.setStyleSheet("")
        match self.__file_type:
            case FileType.ONE_IMAGE:
                img_path = files_paths[0]
                success, data = self.__load_image_path(img_path)
                if not success:
                    return
            case FileType.MANY_IMAGES:
                success, data = self.__load_images_paths(files_paths)
                if not success:
                    return
            case FileType.VIDEO:
                video_path = files_paths[0]
                if not self.__is_video_path_valid(video_path):
                    return
                data = video_path
        self.__send_data(data)

    def __load_image_path(self, image_path: str) -> tuple[bool, ImageNP | None]:
        img = cv2.imread(image_path)
        if img is None:
            error_message("Could not load image")
            return False, None
        return True, img

    def __load_images_paths(self, images_paths: list[str]) -> tuple[bool, list[ImageNP] | None]:
        data = []
        for path in images_paths:
            img = cv2.imread(path)
            if img is None:
                error_message("Could not load some of images")
                return False, None
            data.append(img)
        return True, data

    def __is_video_path_valid(self, video_path: str) -> bool:
        vc = cv2.VideoCapture(video_path)
        ret, frame = vc.read()
        if not ret:
            error_message("Could not load video")
            return False
        return True

    def __send_data(self, data):
        self.file_uploaded.emit(QVariant(data))
        self.accept()
