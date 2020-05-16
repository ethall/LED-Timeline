import os

import cv2 as cv

from _util import Scrubber, get_frames


if not os.path.exists("target_p01_valid.mp4"):
    import p01_extract_valid_frames
color_movie_viewer = Scrubber(
    get_frames(cv.VideoCapture("target_p01_valid.mp4")), window_title="color"
)
color_movie_viewer.create()
color_movie_viewer.wait()


if not os.path.exists("target_p02_denoised.mp4"):
    import p02_denoise
denoised_color_movie_viewer = Scrubber(
    get_frames(cv.VideoCapture("target_p02_denoised.mp4")), window_title="denoised color"
)
denoised_color_movie_viewer.create()
denoised_color_movie_viewer.wait()


if not os.path.exists("target_p03_gray.mp4"):
    import p03_gray
gray_movie_viewer = Scrubber(
    get_frames(cv.VideoCapture("target_p03_gray.mp4")), window_title="gray"
)
gray_movie_viewer.create()
gray_movie_viewer.wait()


if not os.path.exists("target_p04_diff.mp4"):
    import p04_diff
scaled_diff_viewer = Scrubber(
    get_frames(cv.VideoCapture("target_p04_diff.mp4")), window_title="scaled diff"
)
scaled_diff_viewer.create()
scaled_diff_viewer.wait()
