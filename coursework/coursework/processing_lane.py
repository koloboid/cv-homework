import logging
from typing import Optional

import cv2
import numpy as np
from injector import inject
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from coursework.common_types import Box
from coursework.config import Config
from coursework.frame import Frame


logger = logging.getLogger("ProcessingLane")


class ProcessingLane:
    @inject
    def __init__(self, config: Config) -> None:
        self._config = config
        self._debug_mode = config.debug
        self._lower_white = np.array([0, 0, 210], dtype=np.uint8)
        self._upper_white = np.array([180, 30, 255], dtype=np.uint8)
        self._lower_yellow = np.array([15, 80, 150], dtype=np.uint8)
        self._upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        self._y_eval: Optional[np.ndarray] = None

    def handle_frame(self, frame: Frame) -> None:
        height, width = frame.image.shape[:2]
        frame.roi = self._config.lane_crop.multiply(width, height).to_int()
        cropped = frame.image[
            frame.roi.y : frame.roi.bottom,
            frame.roi.x : frame.roi.right,
        ]
        frame.img_cropped_masked = self._filter_white_yellow(cropped)
        gray = frame.img_cropped_masked[:, :, 2]
        frame.img_blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        frame.img_edges = cv2.Canny(frame.img_blurred_gray, 50, 150)

        # frame.img_warped = self._perspective_transform(frame.img_edges)
        self._detect_lines_probabilistic(frame.img_edges, frame)
        if self._y_eval is None:
            self._y_eval = np.linspace(frame.roi.height // 2, frame.roi.height, num=50)
        frame.left_lanes = self._group_lines_ransac(frame.left_lines, frame)
        frame.right_lanes = self._group_lines_ransac(frame.right_lines, frame)

    def _filter_white_yellow(self, img: np.ndarray) -> np.ndarray:
        # convert to HSV and create mask for white colors
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv, self._lower_white, self._upper_white)
        mask_yellow = cv2.inRange(hsv, self._lower_yellow, self._upper_yellow)
        mask = cv2.bitwise_or(mask_white, mask_yellow)
        return cv2.bitwise_and(img, img, mask=mask)

    def _perspective_transform(self, src_img: np.ndarray) -> np.ndarray:
        h, w = src_img.shape[:2]
        src = np.float32(
            [
                [w * 0.51, h * 0.66],
                [w * 0.54, h * 0.66],
                [w * 0.36, h * 0.85],
                [w * 0.66, h * 0.85],
            ],  # type: ignore  # noqa: PGH003
        )
        dst = np.float32(
            [
                [w * 0.36, h * 0.66],
                [w * 0.66, h * 0.66],
                [w * 0.36, h * 0.85],
                [w * 0.66, h * 0.85],
            ],  # type: ignore  # noqa: PGH003
        )
        warp_mat = cv2.getPerspectiveTransform(src, dst)  # type: ignore  # noqa: PGH003
        return cv2.warpPerspective(
            src_img,
            warp_mat,
            (src_img.shape[1], src_img.shape[0]),
        )

    def _detect_lines_probabilistic(self, src: np.ndarray, frame: Frame) -> None:
        lines = cv2.HoughLinesP(
            src,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=20,
            maxLineGap=100,
        )

        if lines is None:
            return

        frame.found_lines = [Box.from_tlrb(line[0]) for line in lines]  # type: ignore  # noqa: PGH003
        for line in frame.found_lines:
            slope = (
                float("inf")
                if line.right - line.x == 0
                else (line.bottom - line.y) / (line.right - line.x)
            )
            if (
                abs(slope) > self._config.lane_min_slope
                and abs(slope) < self._config.lane_max_slope
            ):
                if slope < 0:
                    frame.left_lines.append(line)
                else:
                    frame.right_lines.append(line)

    def _group_lines_ransac(
        self,
        boxes: list[Box],
        frame: Frame,
    ) -> Optional[np.ndarray]:
        if len(boxes) < 2 or self._y_eval is None or frame.roi is None:
            return None
        points = [
            (x, y)
            for box in boxes
            for x, y in [(box.x, box.y), (box.right, box.bottom)]
        ]
        pts = np.array(points)
        xmat = pts[:, 1].reshape(-1, 1)
        ymat = pts[:, 0]

        model = make_pipeline(PolynomialFeatures(1), RANSACRegressor())

        try:
            model.fit(xmat, ymat)
        except Exception:
            logger.exception("Failed to fit lane lines")
            return None

        x_pred = model.predict(self._y_eval.reshape(-1, 1))
        return np.int32(
            np.column_stack((x_pred + frame.roi.x, self._y_eval + frame.roi.y)),  # type: ignore  # noqa: PGH003
        )
