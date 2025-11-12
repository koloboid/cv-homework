from time import monotonic

import cv2
from injector import inject, singleton

from coursework.common_types import Box
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
        if self._config.draw_roi != "original":
            self._draw_roi(frame)
        if self._config.draw_found_lines and frame.found_lines:
            self._draw_lines(frame, frame.found_lines, (128, 128, 128))
        if self._config.draw_filtered_lines:
            self._draw_lines(frame, frame.left_lines, (255, 0, 0))
            self._draw_lines(frame, frame.right_lines, (0, 255, 0))
        self._draw_lanes(frame)
        if self._config.draw_metrics:
            self._draw_metrics(frame)
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
        elif self._config.draw_roi == "warped" and frame.img_warped is not None:
            if len(frame.img_warped.shape) == 2:
                roi_image = cv2.cvtColor(frame.img_warped, cv2.COLOR_GRAY2BGR)
            else:
                roi_image = frame.img_warped

        if roi_image is not None and frame.roi is not None:
            roi = frame.roi
            if roi_image.shape == frame.image.shape:
                frame.image = roi_image
            else:
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

    def _draw_lines(
        self,
        frame: Frame,
        lines: list[Box],
        color: tuple[int, int, int],
    ) -> None:
        if frame.roi is None:
            return
        roi = frame.roi.to_int()
        for line in lines:
            line_int = line.to_int()
            cv2.line(
                frame.image,
                (line_int.x + roi.x, line_int.y + roi.y),
                (line_int.right + roi.x, line_int.bottom + roi.y),
                color,
                2,
            )

    def _draw_lanes(self, frame: Frame) -> None:
        if frame.left_lanes is not None:
            cv2.polylines(frame.image, [frame.left_lanes], False, (0, 0, 255), 5)
        if frame.right_lanes is not None:
            cv2.polylines(frame.image, [frame.right_lanes], False, (0, 0, 255), 5)
