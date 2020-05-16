import cv2 as cv
import numpy as np

from _util import get_frames, matrix_to_video


vid = cv.VideoCapture("target_p03_gray.mp4")

# (t,y,x)
gray_movie = get_frames(vid, grayscale=True)

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
diff_gray_movie = np.diff(np.array(gray_movie, np.int16), axis=0)

print("Removing decreasing and small intensity changes...")
diff_gray_movie[diff_gray_movie < 128] = 0
# Adjust max values to prevent unsafe casting side-effects.
# (Having a value greater than +255 ought to be impossible,
# but better safe than sorry.)
diff_gray_movie[diff_gray_movie > 255] = 255

print("Compressing to grayscale...")
scaled_diff_gray = np.array(diff_gray_movie, np.uint8)

matrix_to_video(
    scaled_diff_gray,
    "target_p04_diff.mp4",
    "mp4v",
    int(vid.get(cv.CAP_PROP_FPS)),
    int(vid.get(cv.CAP_PROP_FRAME_WIDTH)),
    int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)),
    is_color=False
)

