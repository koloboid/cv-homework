from time import monotonic

import cv2
from injector import inject, singleton

from coursework.config import Config
from coursework.frame import Frame


@singleton
class ProcessingDraw:
    @inject
    def __init__(self, config: Config) -> None:
        self._config = config
        self._frame_count = 0
        self._frame_time_sum = 0.0
        self._prev_reset_time = 0.0

    def handle_frame(self, frame: Frame) -> None:
        now = monotonic()
        self._frame_count += 1
        self._frame_time_sum += now - frame.timestamp
        if self._config.draw_metrics:
            self._draw_metrics(frame)
        if self._config.draw_roi != "original":
            self._draw_roi(frame)
        if self._prev_reset_time + 1.0 < now:
            self._frame_count = 0
            self._frame_time_sum = 0.0
            self._prev_reset_time = now

    def _draw_roi(self, frame: Frame) -> None:
        roi_image = None
        if self._config.draw_roi == "edges" and frame.img_edges is not None:
            roi_image = cv2.cvtColor(frame.img_edges, cv2.COLOR_GRAY2BGR)
        elif self._config.draw_roi == "blurred" and frame.img_blurred_gray is not None:
            roi_image = cv2.cvtColor(frame.img_blurred_gray, cv2.COLOR_GRAY2BGR)
        elif self._config.draw_roi == "masked" and frame.img_cropped_masked is not None:
            roi_image = frame.img_cropped_masked

        if roi_image is not None and frame.roi is not None:
            roi = frame.roi
            frame.image[roi.y : roi.bottom, roi.x : roi.right] = roi_image

    def _draw_metrics(self, frame: Frame) -> None:
        avg_fps = self._frame_count / self._frame_time_sum
        avg_ftime = self._frame_time_sum / self._frame_count
        cv2.putText(
            frame.image,
            f"Avg FPS: {avg_fps:.2f}; FTIME: {avg_ftime*1000:.2f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
