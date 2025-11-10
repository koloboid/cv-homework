import logging

import cv2
import numpy as np
from injector import inject

from coursework.config import Config
from coursework.frame import Frame


logger = logging.getLogger("ProcessingLane")


class ProcessingLane:
    @inject
    def __init__(self, config: Config) -> None:
        self._config = config
        self._debug_mode = config.debug

    def handle_frame(self, frame: Frame) -> None:
        height, width = frame.image.shape[:2]
        crop = self._config.lane_crop
        x, y, w, h = (
            int(crop[0] * width),
            int(crop[1] * height),
            int(crop[2] * width),
            int(crop[3] * height),
        )

        cv2.rectangle(frame.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cropped = frame.image[y : y + h, x : x + w]
        hls = cv2.cvtColor(cropped, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, 200, 0], dtype=np.uint8)
        upper_white = np.array([180, 255, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hls, lower_white, upper_white)
        masked = cv2.bitwise_and(cropped, cropped, mask=mask_white)

        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        frame.image[y : y + h, x : x + w] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        filtered_lines = self._detect_lanes_probabilistic(edges)

        if filtered_lines:
            logger.debug(
                f"Found {len(filtered_lines)} lane lines using probabilistic Hough transform",
            )
            for line in filtered_lines:
                x1, y1, x2, y2 = line
                cv2.line(
                    frame.image, (x1 + x, y1 + y), (x2 + x, y2 + y), (0, 0, 255), 3
                )

    def _detect_lanes_probabilistic(self, edges: np.ndarray) -> list:
        min_slope = 0.2
        max_slope = 3.0
        print(edges.shape)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=20,
            maxLineGap=100,
        )

        if lines is None:
            return []

        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            slope = float("inf") if x2 - x1 == 0 else (y2 - y1) / (x2 - x1)
            if abs(slope) > min_slope and abs(slope) < max_slope:
                filtered_lines.append([x1, y1, x2, y2])

        return filtered_lines
