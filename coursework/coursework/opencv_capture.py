import asyncio
import logging
from time import monotonic
from typing import Callable, Optional

import cv2
from injector import inject, singleton

from coursework.config import Config
from coursework.frame import Frame


logger = logging.getLogger("OpenCVCapture")


@singleton
class OpenCVCapture:
    @inject
    def __init__(self, config: Config) -> None:
        self._config = config
        self._frame_callback: Optional[Callable[[Frame], None]] = None
        self.cap: Optional[cv2.VideoCapture] = None

    async def init(self, frame_callback: Callable[[Frame], None]) -> None:
        self._frame_callback = frame_callback
        if self.cap is None:
            self.cap = cv2.VideoCapture(self._config.capture_source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.capture_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.capture_height)
            self.cap.set(cv2.CAP_PROP_FPS, self._config.capture_fps)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera for url {self._config.capture_source}",
            )

    async def run(self) -> None:
        if self.cap is None or self._frame_callback is None:
            raise RuntimeError("Capture not initialized")
        logger.info("Starting capture loop")
        while True:
            ret, image = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from capture")
                continue
            frame = Frame(image=image, timestamp=monotonic())
            self._frame_callback(frame)
            await asyncio.sleep(1 / self._config.capture_fps)

    async def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
