from typing import List, Set

import cv2 as _cv
import numpy as _np

from _util import (
    Config as _Config,
    Rect,
    Timeline,
    get_frames as _get_frames,
)


_cfg = _Config("config.json")


def detect_leds(source: str) -> List[Rect]:
    diff_frames: _np.ndarray

    video = _cv.VideoCapture(source)
    diff_frames = _get_frames(video, grayscale=True)
    video.release()
    del video

    ksize = (
        _cfg.detect.blur.kernel_size.width,
        _cfg.detect.blur.kernel_size.height,
    )
    aperture = _cfg.detect.canny.aperture
    low_threshold = _cfg.detect.canny.low_threshold
    low_threshold_mult = _cfg.detect.canny.low_threshold_multiplier

    print("Detecting LEDs...")
    results: List[Rect] = []
    for i in range(diff_frames.shape[0]):
        blurred = _cv.blur(diff_frames[i], ksize)
        edges = _cv.Canny(
            blurred,
            low_threshold,
            low_threshold * low_threshold_mult,
            apertureSize=aperture,
        )
        contours, _ = _cv.findContours(
            edges, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_SIMPLE
        )
        hulls: List[_np.ndarray] = []
        for c in contours:
            hulls.append(_cv.convexHull(c))
        # Find the indices of hulls that lie inside another hull.
        nested_hull_indices: Set[int] = set()
        if len(hulls) > 1:
            for j, h1 in enumerate(hulls):
                for k, h2 in enumerate(hulls):
                    if h1 is h2:
                        continue
                    area, _ = _cv.intersectConvexConvex(h1, h2, handleNested=True)
                    if area > 0.0:
                        if area == _cv.contourArea(h1):
                            nested_hull_indices.add(j)
                        elif area == _cv.contourArea(h2):
                            nested_hull_indices.add(k)
        # Create bounding rectanges
        for j, h in enumerate(hulls):
            if j in nested_hull_indices:
                continue
            # boundingRect -> (x, y, w, h)
            results.append(Rect(i, *_cv.boundingRect(h)))

    return results


def record_states_to_csv(source: str, destination: str, leds: List[Rect]) -> None:
    timeline = Timeline(source)
    print("Tracking LED state changes...")
    timeline.record(leds, _cfg.diff.minimum_threshold)
    with open(destination, "w+") as f:
        f.write(str(timeline))


if __name__ == "__main__":
    rects = detect_leds("target_p04_diff.mp4")

    if len(rects) == 0:
        print("Failed to find any LEDs")
        exit(1)
    for r in rects:
        print(f"  {r}")

    record_states_to_csv("target_p03_gray.mp4", "target_p05_timeline.csv", rects)
