import typing

import numpy

from .core.types import Size
from .videoio import VideoCaptureAPIs

class VideoWriter:
    @typing.overload
    def __init__(
        self,
        filename: str,
        apiPreference: VideoCaptureAPIs,
        fourcc: int,
        fps: float,
        frameSize: Size,
        isColor: bool = ...,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        filename: str,
        fourcc: int,
        fps: float,
        frameSize: Size,
        isColor: bool = ...,
    ) -> None: ...
    def __init__(self) -> None: ...
    def write(self, image: numpy.ndarray) -> None: ...
    def release(self) -> None: ...

def VideoWriter_fourcc(char1: str, char2: str, char3: str, char4: str) -> int: ...
