import enum
import typing

import numpy

from .core.types import Size

class ColorConversionCodes(enum.Enum):
    COLOR_BGR2GRAY = ...

COLOR_BGR2GRAY = ColorConversionCodes.COLOR_BGR2GRAY

class ContourApproximationModes(enum.Enum):
    CHAIN_APPROX_NONE = ...
    CHAIN_APPROX_SIMPLE = ...
    CHAIN_APPROX_TC89_L1 = ...
    CHAIN_APPROX_TC89_KCOS = ...

CHAIN_APPROX_NONE = ContourApproximationModes.CHAIN_APPROX_NONE
CHAIN_APPROX_SIMPLE = ContourApproximationModes.CHAIN_APPROX_SIMPLE
CHAIN_APPROX_TC89_L1 = ContourApproximationModes.CHAIN_APPROX_TC89_L1
CHAIN_APPROX_TC89_KCOS = ContourApproximationModes.CHAIN_APPROX_TC89_KCOS

class RetrievalModes(enum.Enum):
    RETR_EXTERNAL = ...
    RETR_LIST = ...
    RETR_CCOMP = ...
    RETR_TREE = ...
    RETR_FLOODFILL = ...

RETR_EXTERNAL = RetrievalModes.RETR_EXTERNAL
RETR_LIST = RetrievalModes.RETR_LIST
RETR_CCOMP = RetrievalModes.RETR_CCOMP
RETR_TREE = RetrievalModes.RETR_TREE
RETR_FLOODFILL = RetrievalModes.RETR_FLOODFILL

def blur(src: numpy.ndarray, ksize: Size) -> numpy.ndarray: ...
def boundingRect(array: numpy.ndarray) -> typing.Tuple[int, int, int, int]: ...
def Canny(
    image: numpy.ndarray, threshold1: float, threshold2: float, apertureSize: int = ...
) -> numpy.ndarray: ...
def contourArea(contour: numpy.ndarray) -> float: ...
def convexHull(points: numpy.ndarray) -> numpy.ndarray: ...
def cvtColor(src: numpy.ndarray, code: ColorConversionCodes) -> numpy.ndarray: ...
def findContours(
    image: numpy.ndarray, mode: RetrievalModes, method: ContourApproximationModes
) -> typing.Tuple[typing.List[numpy.ndarray], numpy.ndarray]: ...
def intersectConvexConvex(
    _p1: numpy.ndarray, _p2: numpy.ndarray, handleNested: bool = ...
) -> typing.Tuple[float, numpy.ndarray]: ...
