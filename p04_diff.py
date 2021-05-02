import cv2 as _cv
import numpy as _np

from _util import (
    Config as _Config,
    get_frames as _get_frames,
    matrix_to_video as _matrix_to_video,
)


_cfg = _Config("config.json")


def to_intensity_difference(source: str, destination: str) -> None:
    vid = _cv.VideoCapture(source)

    # (t,y,x)
    gray_movie = _get_frames(vid, grayscale=True)

    """

    Post-diff, per-frame value scale:

                          ΔIntensity/time
    -255          -128            0            128           255
      |-------------|-------------|-------------|-------------|
      |                decrease < | > increase  |             |
    max decrease                  |             |    max increase
      |                       no change         |
      |                                         |
      |------------- coerced to 0 --------------|

    We only care about LEDs that are turning on, so most values
    are dropped.

    """

    print("Calculating pixel intensity over time...")
    # (t,y,x)
    diff_gray_movie = _np.diff(_np.array(gray_movie, _np.int16), axis=0)  # type: ignore

    print("Removing decreasing and small intensity changes...")
    diff_gray_movie[diff_gray_movie < _cfg.diff.minimum_threshold] = 0
    # Adjust max values to prevent unsafe casting side-effects.
    # (Having a value greater than +255 ought to be impossible,
    # but better safe than sorry.)
    diff_gray_movie[diff_gray_movie > 255] = 255

    print("Compressing to grayscale...")
    scaled_diff_gray = _np.array(diff_gray_movie, _np.uint8)  # type: ignore

    _matrix_to_video(
        scaled_diff_gray,
        destination,
        "mp4v",
        int(vid.get(_cv.CAP_PROP_FPS)),
        int(vid.get(_cv.CAP_PROP_FRAME_WIDTH)),
        int(vid.get(_cv.CAP_PROP_FRAME_HEIGHT)),
        is_color=False,
    )


if __name__ == "__main__":
    to_intensity_difference("target_p03_gray.mp4", "target_p04_diff.mp4")
