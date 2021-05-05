import datetime
import enum
import json
from decimal import Decimal, ROUND_DOWN as DECIMAL_ROUND_DOWN
from typing import Any, Dict, List

import cv2 as cv
import numpy as np


class Config:
    class Denoise:
        def __init__(self, **kwargs: int) -> None:
            self.search_window_size = kwargs["searchWindowSize"]
            self.strength = kwargs["strength"]
            self.template_window_size = kwargs["templateWindowSize"]
            self.temporal_window_size = kwargs["temporalWindowSize"]

        def serialize(self) -> Dict[str, int]:
            return {
                "searchWindowSize": self.search_window_size,
                "strength": self.strength,
                "templateWindowSize": self.template_window_size,
                "temporalWindowSize": self.temporal_window_size,
            }

    class Detect:
        class Blur:
            class Kernel:
                def __init__(self, **kwargs: int) -> None:
                    self.width = kwargs["width"]
                    self.height = kwargs["height"]

                def serialize(self) -> Dict[str, int]:
                    return {
                        "width": self.width,
                        "height": self.height,
                    }

            def __init__(self, **kwargs: Any) -> None:
                self.kernel_size = Config.Detect.Blur.Kernel(**kwargs["kernelSize"])

            def serialize(self) -> Dict[str, Any]:
                return {
                    "kernelSize": self.kernel_size.serialize(),
                }

        class Canny:
            def __init__(self, **kwargs: int) -> None:
                self.aperture = kwargs["aperture"]
                self.low_threshold = kwargs["lowThreshold"]
                self.low_threshold_multiplier = kwargs["lowThresholdMultiplier"]

            def serialize(self) -> Dict[str, int]:
                return {
                    "aperture": self.aperture,
                    "lowThreshold": self.low_threshold,
                    "lowThresholdMultiplier": self.low_threshold_multiplier,
                }

        def __init__(self, **kwargs: Any) -> None:
            self.blur = Config.Detect.Blur(**kwargs["blur"])
            self.canny = Config.Detect.Canny(**kwargs["canny"])

        def serialize(self) -> Dict[str, Any]:
            return {
                "blur": self.blur.serialize(),
                "canny": self.canny.serialize(),
            }

    class Diff:
        def __init__(self, **kwargs: int) -> None:
            self.minimum_threshold = kwargs["minimumThreshold"]

        def serialize(self) -> Dict[str, int]:
            return {
                "minimumThreshold": self.minimum_threshold,
            }

    def __new__(cls, file: str) -> "Config":
        singleton = cls.__dict__.get("__singleton__")
        if singleton is not None:
            return singleton
        cls.__singleton__ = super(Config, cls).__new__(cls)
        return cls.__singleton__

    def __init__(self, file: str) -> None:
        import os.path

        self._initial_config: Dict[str, Any] = {}
        self.file = os.path.abspath(file)
        self.denoise: Config.Denoise
        self.detect: Config.Detect
        self.diff: Config.Diff
        self.reload()

    def reload(self) -> None:
        with open(self.file, "r") as config:
            self._initial_config = json.load(config)

        self.denoise = Config.Denoise(**self._initial_config["denoise"])
        self.detect = Config.Detect(**self._initial_config["detect"])
        self.diff = Config.Diff(**self._initial_config["diff"])

    def save(self) -> None:
        with open(self.file, "w") as config:
            json.dump(self.serialize(), config, indent=4)

    def serialize(self) -> Dict[str, Any]:
        return {
            "denoise": self.denoise.serialize(),
            "diff": self.diff.serialize(),
            "detect": self.detect.serialize(),
        }


@enum.unique
class LedState(enum.Enum):
    ON = True
    OFF = False

    @staticmethod
    def _crop(rect: "Rect", grayframe: np.ndarray) -> Any:
        return grayframe[
            rect.y : (rect.y + rect.height), rect.x : (rect.x + rect.width)
        ]

    @staticmethod
    def get_state(
        rect: "Rect", grayframe: np.ndarray, threshold: int, desired_state: "LedState"
    ) -> "LedState":
        """grayframe: np.ndarray(y, x)"""
        frame = rect.crop(grayframe)
        if desired_state == LedState.ON:
            frame[frame < threshold] = 0  # type: ignore
            return LedState.ON if frame.mean() > threshold else LedState.OFF  # type: ignore
        else:
            frame[frame > threshold] = 0  # type: ignore
            return LedState.ON if frame.mean() < threshold else LedState.OFF  # type: ignore


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

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y : (self.y + self.height), self.x : (self.x + self.width)]


class Scrubber:
    def __init__(self, matrix: np.ndarray, window_title: str = "Scrubber"):
        """matrix is some np.array(time,height,width[,chan])"""
        self.matrix = matrix
        self.window_title = window_title
        self.trackbar_name = "frame"

    def _trackbar_callback(self, value: int) -> None:
        cv.imshow(self.window_title, self.matrix[value])

    def create(self):
        cv.namedWindow(self.window_title)
        cv.createTrackbar(
            self.trackbar_name,
            self.window_title,
            0,
            self.matrix.shape[0] - 1,
            self._trackbar_callback,
        )
        self._trackbar_callback(0)

    def wait(self):
        cv.waitKey()


class Timeline:
    def __init__(self, source: str):
        super().__init__()
        self._headers: List[str] = ["frame", "timestamp"]
        self._state_store: Dict[int, List[LedState]] = {}

        video = cv.VideoCapture(source)
        self.grayframes = get_frames(video, grayscale=True)
        self.timestamps = get_frame_times(video)
        video.release()

    def __str__(self) -> str:
        rows: List[str] = [",".join(self._headers)]
        frames = list(self._state_store.keys())
        frames.sort()
        for f, t in zip(frames, self.timestamps):
            cells: List[str] = [str(f)]
            cells += [str(t)]
            cells += [str(int(state.value)) for state in self._state_store[f]]
            rows.append(",".join(cells))
        return "\n".join(rows)

    def record(self, rects: List[Rect], threshold: int) -> None:
        """grayframes: _np.ndarray(t, y, x)"""
        self._headers += [r.name.replace(",", ";") for r in rects]
        for i in range(self.grayframes.shape[0]):
            self._state_store[i] = [
                LedState.get_state(r, self.grayframes[i], threshold, LedState.ON)
                for r in rects
            ]


class Timestamp(datetime.time):
    """ISO-8601 microsecond timestamp format."""

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        args = f"{self.hour}, {self.minute}, {self.second}, {self.microsecond}"
        return f"{classname}({args})"

    def __str__(self) -> str:
        return super().isoformat(timespec="microseconds")


def get_frames(video: cv.VideoCapture, grayscale: bool = False) -> np.ndarray:
    """AssertionError is raised if a frame cannot be read."""
    video.set(cv.CAP_PROP_POS_AVI_RATIO, 0)

    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    shape = (
        (frame_count, height, width) if grayscale else (frame_count, height, width, 3)
    )
    result = np.zeros(shape, np.uint8)  # type: ignore

    for index in range(frame_count):
        success, frame = video.read()
        assert success, f"Error reading frame {index}"
        result[index] = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if grayscale else frame

    return result


def get_frame_times(video: cv.VideoCapture) -> List[Timestamp]:
    def decimal_to_int(value: Decimal) -> int:
        return int(value.to_integral_value(rounding=DECIMAL_ROUND_DOWN))

    period = Decimal(1 / Decimal(video.get(cv.CAP_PROP_FPS)))

    timeline_seconds = [
        Decimal(period * i).quantize(Decimal("0.000001"), rounding=DECIMAL_ROUND_DOWN)
        for i in range(int(video.get(cv.CAP_PROP_FRAME_COUNT)))
    ]

    results: List[Timestamp] = []

    seconds_per_hour = 3600
    seconds_per_minute = 60
    for seconds in timeline_seconds:
        remaining_seconds = decimal_to_int(seconds)

        hours = remaining_seconds // seconds_per_hour
        remaining_seconds -= seconds_per_hour * hours

        minutes = remaining_seconds // seconds_per_minute
        remaining_seconds -= seconds_per_minute * minutes

        microseconds = seconds - int(seconds)
        microseconds = microseconds.shift(
            abs(microseconds.as_tuple().exponent)
        ).normalize()

        results.append(
            Timestamp(hours, minutes, remaining_seconds, decimal_to_int(microseconds))
        )

    return results


def matrix_to_video(
    matrix: np.ndarray,
    filepath: str,
    codec: str,
    fps: float,
    width: int,
    height: int,
    is_color: bool = True,
) -> None:
    _codec = cv.VideoWriter_fourcc(*codec)

    output = cv.VideoWriter(filepath, _codec, fps, (width, height), isColor=is_color)

    for index in range(matrix.shape[0]):
        output.write(matrix[index])

    output.release()
