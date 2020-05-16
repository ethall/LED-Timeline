from typing import Optional

import cv2 as cv
import numpy as np


class Scrubber:
    def __init__(self, matrix: np.ndarray, window_title: str = "Scrubber"):
        """matrix is some np.array(time,height,width[,chan])"""
        self.matrix = matrix
        self.window_title = window_title
        self.trackbar_name = "frame"

    def _trackbar_callback(self, value):
        cv.imshow(self.window_title, self.matrix[value])

    def create(self):
        cv.namedWindow(self.window_title)
        cv.createTrackbar(
            self.trackbar_name,
            self.window_title,
            0,
            self.matrix.shape[0] - 1,
            self._trackbar_callback
        )
        self._trackbar_callback(0)

    def wait(self):
        cv.waitKey()


def get_frames(
    video: cv.VideoCapture,
    grayscale: bool = False
) -> np.ndarray:
    """AssertionError is raised if a frame cannot be read."""
    video.set(cv.CAP_PROP_POS_AVI_RATIO, 0)

    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    result = np.zeros(
        (frame_count, height, width) if grayscale else (frame_count, height, width, 3),
        np.uint8
    )

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
    is_color: bool = True
) -> None:
    """`frame_count` defaults to matrix.shape[0] if None."""
    _codec = cv.VideoWriter_fourcc(*codec)

    output = cv.VideoWriter(filepath, _codec, fps, (width, height), isColor=is_color)

    if frame_count is None:
        frame_count = matrix.shape[0]

    for index in range(frame_count):
        output.write(matrix[index])

    output.release()
