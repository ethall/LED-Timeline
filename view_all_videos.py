import os

import cv2 as cv

from _util import Scrubber, get_frames
from p01_extract_valid_frames import extract_valid_frames
from p02_denoise import denoise
from p03_gray import to_grayscale
from p04_diff import to_intensity_difference


if not os.path.exists("target_p01_valid.mp4"):
    extract_valid_frames("target.mp4", "target_p01_valid.mp4")
color_movie_viewer = Scrubber(
    get_frames(cv.VideoCapture("target_p01_valid.mp4")), window_title="color"
)
color_movie_viewer.create()
color_movie_viewer.wait()


if not os.path.exists("target_p02_denoised.mp4"):
    denoise("target_p01_valid.mp4", "target_p02_denoised.mp4")
denoised_color_movie_viewer = Scrubber(
    get_frames(cv.VideoCapture("target_p02_denoised.mp4")), window_title="denoised color"
)
denoised_color_movie_viewer.create()
denoised_color_movie_viewer.wait()


if not os.path.exists("target_p03_gray.mp4"):
    to_grayscale("target_p02_denoised.mp4", "target_p03_gray.mp4")
gray_movie_viewer = Scrubber(
    get_frames(cv.VideoCapture("target_p03_gray.mp4")), window_title="gray"
)
gray_movie_viewer.create()
gray_movie_viewer.wait()


if not os.path.exists("target_p04_diff.mp4"):
    to_intensity_difference("target_p03_gray.mp4", "target_p04_diff.mp4")
scaled_diff_viewer = Scrubber(
    get_frames(cv.VideoCapture("target_p04_diff.mp4")), window_title="scaled diff"
)
scaled_diff_viewer.create()
scaled_diff_viewer.wait()


if not os.path.exists("target_p05_detect.mp4"):
    print("Skipping part 5: video does not exist")
    print("It can be created by running the 'p05_detect.py' script")
else:
    gray_movie_viewer = Scrubber(
        get_frames(cv.VideoCapture("target_p05_detect.mp4")), window_title="detect"
    )
    gray_movie_viewer.create()
    gray_movie_viewer.wait()
