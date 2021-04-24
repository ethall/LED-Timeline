import argparse
import random as rng  # Used for colors.
from typing import List, Set

import cv2 as cv
import numpy as np


rng.seed(12345)  # Used for colors.

window_name = "Edge Map"
canny_trackbar = "min threshold"
frame_trackbar = "frame"
polys_trackbar = "poly epsilon"


def CannyThreshold(val: int) -> None:
    # Get frame
    frame_index = cv.getTrackbarPos(frame_trackbar, window_name)
    vid.set(cv.CAP_PROP_POS_FRAMES, frame_index)
    _, src = vid.read()
    # Preprocess frame
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    img_blur = cv.blur(src_gray, (3, 3))
    # Get edges using Canny
    low_threshold = cv.getTrackbarPos(canny_trackbar, window_name)
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * 3, 3)
    # Postprocess frame (hide everything except Canny edges)
    mask: np.ndarray = detected_edges != 0
    dst: np.ndarray = src * (mask[:, :, None].astype(src.dtype))  # type: ignore
    cv.imshow(window_name, dst)

    # Calculate outline/external contours
    contours, _ = cv.findContours(
        detected_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    # Analyze contour data
    polys: List[np.ndarray] = []
    all_hulls: List[np.ndarray] = []  # Includes nested hulls.
    for c in contours:
        # Create a polygon from this contour.
        polys.append(
            cv.approxPolyDP(c, cv.getTrackbarPos(polys_trackbar, window_name), True)
        )
        # A polygon is said to be "convex" if every outside angle created from the
        # intersection of its line-segments is greater than 180 degrees.
        # A "hull" is a convex polygon that surrounds another, usually concave, polygon.
        # Hulls have the smallest area possible while maintaining their convex shape.
        all_hulls.append(cv.convexHull(c))
    # Find the indices of hulls that lie inside another hull.
    nested_hull_indices: Set[int] = set()
    if len(all_hulls) > 1:
        for i, h1 in enumerate(all_hulls):
            for j, h2 in enumerate(all_hulls):
                if h1 is h2:
                    continue
                area, _ = cv.intersectConvexConvex(h1, h2, handleNested=True)
                if area > 0.0:
                    if area == cv.contourArea(h1):
                        nested_hull_indices.add(i)
                    elif area == cv.contourArea(h2):
                        nested_hull_indices.add(j)
    # Construct a list of all outer-most hulls.
    hulls: List[np.ndarray] = [
        h for i, h in enumerate(all_hulls) if i not in nested_hull_indices
    ]

    # Draw contours, hulls, and bounding rectangles in another window.
    drawing = src.copy()
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(drawing, polys, i, color)
    for j in range(len(hulls)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(drawing, hulls, j, color)
        rect = cv.boundingRect(hulls[j])
        cv.rectangle(
            drawing,
            (int(rect[0]), int(rect[1])),
            (int(rect[0] + rect[2]), int(rect[1] + rect[3])),
            color,
            2,
        )
    cv.imshow("Contours", drawing)


parser = argparse.ArgumentParser()
parser.add_argument("input", help="Filepath to input video.", metavar="VIDEO")
args = parser.parse_args()

vid = cv.VideoCapture(args.input)

cv.namedWindow(window_name)
cv.createTrackbar(
    frame_trackbar,
    window_name,
    0,
    int(vid.get(cv.CAP_PROP_FRAME_COUNT)) - 1,
    CannyThreshold,
)
cv.createTrackbar(canny_trackbar, window_name, 0, 100, CannyThreshold)
cv.createTrackbar(polys_trackbar, window_name, 1, 20, CannyThreshold)

cv.setTrackbarMin(polys_trackbar, window_name, 1)
cv.setTrackbarPos(polys_trackbar, window_name, 3)
cv.setTrackbarPos(
    canny_trackbar, window_name, 50
)  # This seems to be a nice default (for test_02 anyway)

CannyThreshold(0)
cv.waitKey()
