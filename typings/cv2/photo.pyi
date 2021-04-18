import numpy

def fastNlMeansDenoisingMulti(
    srcImgs: numpy.ndarray,
    imgToDenoiseIndex: int,
    temporalWindowSize: int,
    h: float = ...,
    templateWindowSize: int = ...,
    searchWindowSize: int = ...,
) -> numpy.ndarray: ...
