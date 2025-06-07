from PyQt5.QtCore import QTimer

from Elements.ImageGetters import Video, VideoFile
from GUI.widgets.ImageLabel import ImageLabel


class VideoLabel(ImageLabel):
    """
    Image label that shows video
    """

    _video: Video
    _update_image_timer: QTimer  # Timer used update image

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._video = VideoFile()
        self.update_image()

        self._update_image_timer = QTimer()
        self._update_image_timer.timeout.connect(self.update_image)

    def set_video(
            self,
            video: Video,
    ) -> None:
        self._video = video
        fps = video.fps()
        msec = 1000 // fps
        self._update_image_timer.stop()
        self._update_image_timer.start(msec)

    def update_image(self):
        self.set_image(self._video.get_image())
