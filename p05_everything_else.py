from enum import Enum, unique
from typing import Dict, List, NamedTuple, Tuple

import cv2 as cv
import numpy as np

from _util import get_frames


CANNY_LOW_THRESHOLD = 50
LED_ON_LOW_THRESHOLD = 128


class Rect:
    frame: int
    x: int
    y: int
    width: int
    height: int

    def __init__(self, frame: int, x: int, y: int, width: int, height: int):
        super().__init__()
        self.frame = frame
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __str__(self) -> str:
        return f"{self.name.replace(',', ', ')} @ frame {self.frame}"

    @property
    def name(self) -> str:
        return f"({self.x}x,{self.y}y,{self.width}w,{self.height}h)"


@unique
class LedState(Enum):
    ON = True
    OFF = False

    @staticmethod
    def get_state(rect: Rect, grayframe: np.ndarray) -> "LedState":
        """grayframe: np.ndarray(y, x)"""
        frame = grayframe[rect.y:(rect.y + rect.height), rect.x:(rect.x + rect.width)]
        frame[frame < LED_ON_LOW_THRESHOLD] = 0
        return LedState.ON if frame.mean() > LED_ON_LOW_THRESHOLD else LedState.OFF


class Timeline:
    _headers: List[str]
    _store: Dict[int, List[LedState]]

    def __init__(self):
        super().__init__()
        self._headers = ["frame"]
        self._store = {}

    def __str__(self) -> str:
        rows: List[str] = [",".join(self._headers)]
        frames = list(self._store.keys())
        frames.sort()
        for k in frames:
            cells: List[str] = [f"{k}"]
            cells += [str(int(state.value)) for state in self._store[k]]
            rows.append(",".join(cells))
        return "\n".join(rows)

    def record(self, rects: List[Rect], grayframes: np.ndarray) -> None:
        """grayframes: np.ndarray(t, y, x)"""
        self._headers += [r.name.replace(",", ";") for r in rects]
        for i in range(grayframes.shape[0]):
            self._store[i] = [LedState.get_state(r, grayframes[i]) for r in rects]


diff_frames = get_frames(cv.VideoCapture("target_p04_diff.mp4"), grayscale=True)

print("Detecting LEDs...")
rects: List[Rect] = []
for i in range(diff_frames.shape[0]):
    blurred = cv.blur(diff_frames[i], (3,3))
    edges = cv.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_LOW_THRESHOLD * 3, 3)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    hulls = []
    for c in contours:
        hulls.append(cv.convexHull(c))
    # Find the indices of hulls that lie inside another hull.
    nested_hull_indices = set()
    if len(hulls) > 1:
        for j, h1 in enumerate(hulls):
            for k, h2 in enumerate(hulls):
                if h1 is h2:
                    continue
                area, _ = cv.intersectConvexConvex(h1, h2, handleNested=True)
                if area > 0.0:
                    if area == cv.contourArea(h1):
                        nested_hull_indices.add(j)
                    elif area == cv.contourArea(h2):
                        nested_hull_indices.add(k)
    # Create bounding rectanges
    for j, h in enumerate(hulls):
        if j in nested_hull_indices:
            continue
        # boundingRect -> (x, y, w, h)
        rects.append(Rect(i, *cv.boundingRect(h)))


if len(rects) == 0:
    print("Failed to find any LEDs")
    exit(1)
for r in rects:
    print(f"  {r}")


grayframes = get_frames(cv.VideoCapture("target_p03_gray.mp4"), grayscale=True)
timeline = Timeline()
print("Tracking LED state changes...")
timeline.record(rects, grayframes)
with open("target_p05_timeline.csv", "w+") as f:
    f.write(str(timeline))
