import logging

import cv2
import numpy as np
from injector import inject

from coursework.common_types import Box
from coursework.config import Config
from coursework.frame import Frame


logger = logging.getLogger("ProcessingLane")


class ProcessingLane:
    @inject
    def __init__(self, config: Config) -> None:
        self._config = config
        self._debug_mode = config.debug
        self._lower_white = np.array([0, 210, 0], dtype=np.uint8)
        self._upper_white = np.array([180, 255, 255], dtype=np.uint8)

    def handle_frame(self, frame: Frame) -> None:
        height, width = frame.image.shape[:2]
        frame.roi = self._config.lane_crop.multiply(width, height).to_int()
        cropped = frame.image[
            frame.roi.y : frame.roi.bottom,
            frame.roi.x : frame.roi.right,
        ]
        # convert to HLS and create mask for white colors
        cropped_hls = cv2.cvtColor(cropped, cv2.COLOR_BGR2HLS)
        mask_white = cv2.inRange(cropped_hls, self._lower_white, self._upper_white)
        frame.img_cropped_masked = cv2.bitwise_and(cropped, cropped, mask=mask_white)

        gray = frame.img_cropped_masked[:, :, 1]
        frame.img_blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        frame.img_edges = cv2.Canny(frame.img_blurred_gray, 50, 150)
        self._detect_lanes_probabilistic(frame)

    def _detect_lanes_probabilistic(self, frame: Frame) -> None:
        if frame.img_edges is None:
            return

        min_slope = 0.2
        max_slope = 3.0

        lines = cv2.HoughLinesP(
            frame.img_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=20,
            maxLineGap=100,
        )

        if lines is None:
            return

        frame.found_lines = [Box.from_tlrb(line[0]) for line in lines]  # type: ignore  # noqa: PGH003
        for line in frame.found_lines:
            x1, y1, x2, y2 = line

            slope = float("inf") if x2 - x1 == 0 else (y2 - y1) / (x2 - x1)
            if abs(slope) > min_slope and abs(slope) < max_slope:
                if slope < 0:
                    frame.left_lines.append(line)
                else:
                    frame.right_lines.append(line)
