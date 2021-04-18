import cv2 as _cv
import numpy as _np

from _util import get_frames as _get_frames, matrix_to_video as _matrix_to_video


def to_grayscale(source: str, destination: str) -> None:
    vid = _cv.VideoCapture(source)

    # (t,y,x,chan)
    denoised_color_movie = _get_frames(vid)
    print("Converting to grayscale...")
    # (t,y,x)
    gray_movie = _np.zeros(denoised_color_movie.shape[0:3], _np.uint8)  # type: ignore
    for index in range(denoised_color_movie.shape[0]):
        print(f"  {index + 1: 5}/{denoised_color_movie.shape[0]}\u000D", end="")
        gray_movie[index] = _cv.cvtColor(
            denoised_color_movie[index], _cv.COLOR_BGR2GRAY
        )
    else:
        print("")

    _matrix_to_video(
        gray_movie,
        destination,
        "mp4v",
        int(vid.get(_cv.CAP_PROP_FPS)),
        int(vid.get(_cv.CAP_PROP_FRAME_WIDTH)),
        int(vid.get(_cv.CAP_PROP_FRAME_HEIGHT)),
        is_color=False,
    )


if __name__ == "__main__":
    to_grayscale("target_p02_denoised.mp4", "target_p03_gray.mp4")
