from PyQt5.QtCore import pyqtSignal, QThread, QObject, pyqtSlot
from PyQt5.QtWidgets import QDialog
from GUI.UI.UI_ProgressWorker import Ui_progressWorker


class Worker(QObject):
    iteration_ended = pyqtSignal(int)  # emits iteration number
    work_done = pyqtSignal(object)  # emits final return value
    __stop_flag: bool
    __finished_flag: bool
    __iteration: int
    __yield_value: ...

    def __init__(self, worker_func):
        super().__init__()
        self.__worker_func = worker_func
        self.__iteration = -1
        self.__yield_value = None
        self.__stop_flag = False
        self.__finished_flag = False

    def run(self):
        while not self.__stop_flag:
            try:
                self.__yield_value = next(self.__worker_func)
                self.__iteration += 1
                self.iteration_ended.emit(self.__iteration)
            except StopIteration:
                self.work_done.emit(self.__yield_value)
                self.stop()
        self.__finished_flag = True

    def stop(self):
        self.__stop_flag = True


class ProgressBarDialog(QDialog):
    """
    Dialog window with progress bar of some custom process that is run in separate thread

    Example:

        def pbar_func():
            data = []
            for i in range(420):
                data.append(i)
                yield
            yield data

        def on_finish(result_data):
            print(result_data)

        pbar = ProgressBarDialog(
            total=total,
            worker_func=pbar_func()
        )
        pbar.work_done.connect(on_finish)
        pbar.setModal(True)
        pbar.exec()
    """

    __total: int
    work_done = pyqtSignal(object)  # sends result when the work is finished
    __worker_thread: QThread
    __worker: Worker

    def __init__(self, total: int, worker_func, *args, **kwargs):
        """
        Parameters
        ----------
        total - total number of iterations
        worker_func - generator that yields each iteration and last yield is its return value
        """
        super().__init__(*args, **kwargs)
        self.ui = Ui_progressWorker()
        self.ui.setupUi(self)
        self.__total = total

        self.__worker_thread = QThread()
        self.__worker = Worker(worker_func)
        self.__worker.iteration_ended.connect(self.on_iteration_ended)
        self.__worker.work_done.connect(self.on_worker_finished)
        self.__worker.moveToThread(self.__worker_thread)
        self.__worker_thread.started.connect(self.__worker.run)

    @pyqtSlot(int)
    def on_iteration_ended(self, i: int):
        text = f"{i + 1} of {self.__total}"
        self.ui.label_progress.setText(text)
        percent = int(i / self.__total * 100)
        self.ui.progressBar.setValue(percent)

    @pyqtSlot(object)
    def on_worker_finished(self, result):
        self.work_done.emit(result)
        self.close()

    def exec(self):
        self.__worker_thread.start()
        super().exec()

    def close(self):
        self.__worker.stop()
        self.__worker_thread.quit()
        self.__worker_thread.wait()
        super().close()
