import time
import cv2
import numpy as np
from typing import Callable
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [15, 10]


def show_image(img: np.ndarray, title: str) -> None:
    plt.axis(False)
    plt.imshow(img)
    plt.title(title)
    plt.show()


### Load and show original image


img_orig = cv2.imread("frame.png")
if img_orig is None:
    raise FileNotFoundError("Image 'frame.png' not found.")
img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
show_image(img_rgb, title="Original RGB Image")


### Color rearrangements collage


red, green, blue = cv2.split(img_rgb)
img_rbg = cv2.merge([red, blue, green])
img_grb = cv2.merge([green, red, blue])
img_bgr = cv2.merge([blue, green, red])

out1 = np.hstack([img_rgb, img_rbg])
out2 = np.hstack([img_grb, img_bgr])
out = np.vstack([out1, out2])

show_image(out, title="Color Channel Rearrangements")


### Flipped images collage


def run_and_measure(
    func_name: str, times: int, func: Callable[[], np.ndarray]
) -> np.ndarray:
    start = time.perf_counter()
    result = None
    for _ in range(times):
        result = func()
    end = time.perf_counter()
    if result is None:
        raise ValueError("Function did not return any result.")
    total_time = (end - start) / times
    print(f"x{times} Execution time for {func_name}: {total_time:.4f} seconds")
    return result


def make_collage_naive(images: list[np.ndarray]) -> np.ndarray:
    if len(images) != 4:
        raise ValueError("Expected exactly 4 images for collage.")
    top_row = np.hstack([images[0], np.fliplr(images[1])])
    bottom_row = np.hstack([np.flipud(images[2]), np.flip(images[3], axis=(0, 1))])
    return np.vstack([top_row, bottom_row])


def make_collage_optimized(images: list[np.ndarray]) -> np.ndarray:
    """
    This function is much faster than the naive implementation.
    Requires all images same size and number of channels.
    """
    if len(images) != 4:
        raise ValueError("Expected exactly 4 images for collage.")
    shapes = [img.shape for img in images]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(f"All images must have the same shape, got: {shapes}")

    h, w = shapes[0][:2]
    result = np.empty((h * 2, w * 2, 3), dtype=np.uint8)

    result[:h, :w] = images[0]
    cv2.flip(images[1], 1, result[:h, w:])
    cv2.flip(images[2], 0, result[h:, :w])
    cv2.flip(images[3], -1, result[h:, w:])

    return result


images = [img_rgb, img_rbg, img_grb, img_bgr]

result_cv = run_and_measure(
    "make_collage_optimized",
    1000,
    lambda: make_collage_optimized(images),
)
show_image(result_cv, title="Optimized collage")

result_np = run_and_measure(
    "make_collage_naive",
    1000,
    lambda: make_collage_naive(images),
)
show_image(result_np, title="Naive collage")
