from typing import List

from _util import Config as _Config, Rect, Timeline, get_frames as _get_frames
from p05_detect import detect_leds as _detect_leds


_cfg = _Config("config.json")


def record_states_to_csv(source: str, destination: str, leds: List[Rect]) -> None:
    timeline = Timeline(source)
    print("Tracking LED state changes...")
    timeline.record(leds, _cfg.diff.minimum_threshold)
    with open(destination, "w+") as f:
        f.write(str(timeline))


if __name__ == "__main__":
    import cv2 as cv
    import numpy as np

    video = cv.VideoCapture("target_p04_diff.mp4")
    diff_frames: np.ndarray = _get_frames(video, grayscale=True)
    video.release()
    del video

    rectangles: List[Rect] = []
    for i in range(diff_frames.shape[0]):
        rects, _, _ = _detect_leds(
            diff_frames[i],
            (
                _cfg.detect.blur.kernel_size.width,
                _cfg.detect.blur.kernel_size.height,
            ),
            _cfg.detect.canny.aperture,
            _cfg.detect.canny.low_threshold,
            _cfg.detect.canny.low_threshold_multiplier,
        )
        rectangles += [*rects]

    if len(rectangles) == 0:
        print("Failed to find any LEDs")
        exit(1)
    for r in rectangles:
        print(f"  {r}")

    record_states_to_csv("target_p03_gray.mp4", "target_p05_timeline.csv", rectangles)
