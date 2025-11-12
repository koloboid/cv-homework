from typing import Optional

import numpy as np

from coursework.common_types import Box


class Frame:
    def __init__(self, image: np.ndarray, timestamp: float) -> None:
        self.image = image
        self.timestamp = timestamp
        self.roi: Optional[Box[int]] = None
        self.img_cropped_masked: Optional[np.ndarray] = None
        self.img_blurred_gray: Optional[np.ndarray] = None
        self.img_edges: Optional[np.ndarray] = None
        self.img_warped: Optional[np.ndarray] = None
        self.found_lines: list[Box[int]] = []
        self.left_lines: list[Box[int]] = []
        self.right_lines: list[Box[int]] = []
        self.left_lanes: Optional[np.ndarray] = None
        self.right_lanes: Optional[np.ndarray] = None
