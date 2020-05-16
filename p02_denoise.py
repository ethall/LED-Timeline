import cv2 as cv
import numpy as np

from _util import get_frames, matrix_to_video


vid = cv.VideoCapture("target_p01_valid.mp4")

# (t,y,x,chan)
color_movie = get_frames(vid)

print("Denoising...")
# (t,y,x,chan)
denoised_color_movie = np.zeros(color_movie.shape, color_movie.dtype)
temporalWindowSize = 3
for index in range(color_movie.shape[0]):
    print(f"  {index + 1: 5}/{color_movie.shape[0]}\u000D", end="")
    if (
        index < temporalWindowSize // 2
        or (color_movie.shape[0] - 1) - index < temporalWindowSize // 2
    ):
        denoised_color_movie[index] = color_movie[index]
        continue
    denoised_color_movie[index] = cv.fastNlMeansDenoisingMulti(
        color_movie, index, temporalWindowSize, searchWindowSize=15
    )
else:
    print("")

# Save results
matrix_to_video(
    denoised_color_movie,
    "target_p02_denoised.mp4",
    "mp4v",
    int(vid.get(cv.CAP_PROP_FPS)),
    int(vid.get(cv.CAP_PROP_FRAME_WIDTH)),
    int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
)
