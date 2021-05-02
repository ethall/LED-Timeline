import json
from typing import Any, Dict, Optional

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


def matrix_to_video(
    matrix: np.ndarray,
    filepath: str,
    codec: str,
    fps: int,
    width: int,
    height: int,
    frame_count: Optional[int] = None,
    is_color: bool = True,
) -> None:
    """`frame_count` defaults to matrix.shape[0] if None."""
    _codec = cv.VideoWriter_fourcc(*codec)

    output = cv.VideoWriter(filepath, _codec, fps, (width, height), isColor=is_color)

    if frame_count is None:
        frame_count = matrix.shape[0]

    for index in range(frame_count):
        output.write(matrix[index])

    output.release()
