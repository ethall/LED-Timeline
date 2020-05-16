import cv2 as cv
import numpy as np

from _util import get_frames, matrix_to_video


vid = cv.VideoCapture("target_p02_denoised.mp4")

# (t,y,x,chan)
denoised_color_movie = get_frames(vid)
print("Converting to grayscale...")
# (t,y,x)
gray_movie = np.zeros(denoised_color_movie.shape[0:3], np.uint8)
for index in range(denoised_color_movie.shape[0]):
    print(f"  {index + 1: 5}/{denoised_color_movie.shape[0]}\u000D", end="")
    gray_movie[index] = cv.cvtColor(denoised_color_movie[index], cv.COLOR_BGR2GRAY)
else:
    print("")

matrix_to_video(
    gray_movie,
    "target_p03_gray.mp4",
    "mp4v",
    int(vid.get(cv.CAP_PROP_FPS)),
    int(vid.get(cv.CAP_PROP_FRAME_WIDTH)),
    int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)),
    is_color=False
)
