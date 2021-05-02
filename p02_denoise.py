import cv2 as _cv
import numpy as _np

from _util import (
    Config as _Config,
    get_frames as _get_frames,
    matrix_to_video as _matrix_to_video,
)


_cfg = _Config("config.json")


def denoise(source: str, destination: str) -> None:
    vid = _cv.VideoCapture(source)

    # (t,y,x,chan)
    color_movie = _get_frames(vid)

    temporal_win_size = _cfg.denoise.temporal_window_size
    strength = _cfg.denoise.strength
    template_win_size = _cfg.denoise.template_window_size
    search_win_size = _cfg.denoise.search_window_size

    print("Denoising...")
    # (t,y,x,chan)
    denoised_color_movie = _np.zeros(color_movie.shape, color_movie.dtype)  # type: ignore
    for index in range(color_movie.shape[0]):
        print(f"  {index + 1: 5}/{color_movie.shape[0]}\u000D", end="")
        if (
            index < temporal_win_size // 2
            or (color_movie.shape[0] - 1) - index < temporal_win_size // 2
        ):
            denoised_color_movie[index] = color_movie[index]
            continue
        denoised_color_movie[index] = _cv.fastNlMeansDenoisingMulti(
            color_movie,
            index,
            temporal_win_size,
            h=strength,
            templateWindowSize=template_win_size,
            searchWindowSize=search_win_size,
        )
    else:
        print("")

    # Save results
    _matrix_to_video(
        denoised_color_movie,
        destination,
        "mp4v",
        int(vid.get(_cv.CAP_PROP_FPS)),
        int(vid.get(_cv.CAP_PROP_FRAME_WIDTH)),
        int(vid.get(_cv.CAP_PROP_FRAME_HEIGHT)),
    )


if __name__ == "__main__":
    denoise("target_p01_valid.mp4", "target_p02_denoised.mp4")
