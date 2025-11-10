import numpy as np


class Frame:
    def __init__(self, image: np.ndarray, timestamp: float):
        self.image = image
        self.timestamp = timestamp
