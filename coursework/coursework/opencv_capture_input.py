from typing import Optional
import cv2

from coursework.config import Config


class OpenCVCaptureInput:
    def __init__(self, config: Config) -> None:
        self._config = config
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self._config.input_source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera for url {self._config.input_source}"
            )

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
