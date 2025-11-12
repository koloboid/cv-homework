import logging

import cv2
from injector import inject, singleton

from coursework.config import Config
from coursework.frame import Frame


logger = logging.getLogger("OpenCVDisplayOutput")


@singleton
class OpenCVDisplayOutput:
    @inject
    def __init__(self, config: Config) -> None:
        self._config = config

    def handle_frame(self, frame: Frame) -> None:
        cv2.imshow("Output", frame.image)
        key = cv2.waitKey(1)
        if key == ord("q"):
            logger.info("Quit signal received, stopping.")
            raise KeyboardInterrupt
        if key == ord("d"):
            self._config.debug = not self._config.debug
            logger.info(f"Debug mode set to {self._config.debug}")
        elif key == ord("0"):
            self._config.draw_roi = "original"
            logger.info(f"Draw ROI set to {self._config.draw_roi}")
        elif key == ord("3"):
            self._config.draw_roi = "edges"
            logger.info(f"Draw ROI set to {self._config.draw_roi}")
        elif key == ord("2"):
            self._config.draw_roi = "blurred"
            logger.info(f"Draw ROI set to {self._config.draw_roi}")
        elif key == ord("1"):
            self._config.draw_roi = "masked"
            logger.info(f"Draw ROI set to {self._config.draw_roi}")

    async def init(self) -> None:
        pass

    async def stop(self) -> None:
        pass
