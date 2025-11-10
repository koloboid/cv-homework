import logging

import cv2
from injector import singleton

from coursework.frame import Frame


logger = logging.getLogger("OpenCVDisplayOutput")


@singleton
class OpenCVDisplayOutput:
    def handle_frame(self, frame: Frame) -> None:
        cv2.imshow("Output", frame.image)
        cv2.waitKey(1)

    async def init(self) -> None:
        pass

    async def stop(self) -> None:
        pass
