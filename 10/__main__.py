import time
from typing import Callable
import cv2

cap = cv2.VideoCapture(0)

box_size = (100, 100)
use_grayscale = False

tracking_descriptors: list[
    tuple[tuple[int, int, int], Callable[[], cv2.Tracker], bool]
] = [
    ((255, 0, 0), cv2.TrackerMIL.create, True),
    ((0, 255, 0), cv2.TrackerKCF.create, False),  # KCF does not support grayscale
    ((0, 0, 255), cv2.TrackerCSRT.create, True),
]
trackers: list[cv2.Tracker | None] = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at coordinates: ({x}, {y})")
        trackers.clear()
        for _, tracker_creator, _ in tracking_descriptors:
            tracker = tracker_creator()
            tracker.init(
                frame,
                (x - box_size[0] // 2, y - box_size[1] // 2, box_size[0], box_size[1]),
            )
            trackers.append(tracker)


callback_set = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    grayscale_frame = None
    for idx, tracker in enumerate(trackers):
        if tracker is None:
            continue
        start_at = time.perf_counter()

        if use_grayscale:
            if not tracking_descriptors[idx][2]:
                continue
            if grayscale_frame is None:
                grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tracking_frame = grayscale_frame
        else:
            tracking_frame = frame
        success, box = tracker.update(tracking_frame)
        elapsed = (time.perf_counter() - start_at) * 1000
        if success:
            color = tracking_descriptors[idx % len(tracking_descriptors)][0]
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, color, 2, 1)
            cv2.putText(
                frame,
                f"{tracker.__class__.__name__} {elapsed:.1f}ms",
                (25, idx * 30 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        else:
            trackers[idx] = None
            print(f"Tracker {tracker.__class__.__name__} lost")

    cv2.imshow("Camera Feed", frame)
    if not callback_set:
        cv2.setMouseCallback("Camera Feed", mouse_callback)
        callback_set = True

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
