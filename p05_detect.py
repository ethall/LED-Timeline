from typing import ClassVar, List, NamedTuple, Optional, Set, Tuple

import cv2 as _cv
import numpy as _np

from _util import (
    Config as _Config,
    Rect,
    get_frames as _get_frames,
)


class LedDetectionResults(NamedTuple):
    rectangles: List[Rect]
    contours: List[_np.ndarray]
    hulls: List[_np.ndarray]


def detect_leds(
    frame: _np.ndarray,
    ksize: Tuple[int, int],
    aperture: int,
    low_threshold: int,
    low_threshold_mult: int,
) -> LedDetectionResults:
    rectangles: List[Rect] = []

    blurred = _cv.blur(frame, ksize)
    edges = _cv.Canny(
        blurred,
        low_threshold,
        low_threshold * low_threshold_mult,
        apertureSize=aperture,
    )
    contours, _ = _cv.findContours(edges, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_SIMPLE)
    hulls: List[_np.ndarray] = []
    for c in contours:
        hulls.append(_cv.convexHull(c))
    # Find the indices of hulls that lie inside another hull.
    nested_hull_indices: Set[int] = set()
    if len(hulls) > 1:
        for j, h1 in enumerate(hulls):
            for k, h2 in enumerate(hulls):
                if h1 is h2:
                    continue
                area, _ = _cv.intersectConvexConvex(h1, h2, handleNested=True)
                if area > 0.0:
                    if area == _cv.contourArea(h1):
                        nested_hull_indices.add(j)
                    elif area == _cv.contourArea(h2):
                        nested_hull_indices.add(k)
    # Create bounding rectanges
    for j, h in enumerate(hulls):
        if j in nested_hull_indices:
            continue
        # boundingRect -> (x, y, w, h)
        rectangles.append(Rect(*_cv.boundingRect(h)))

    return LedDetectionResults(rectangles, contours, hulls)


class _DisplayManager:
    aperture_trackbar: ClassVar[str] = "Aperture"
    display_contours_trackbar: ClassVar[str] = "Contours"
    display_hulls_trackbar: ClassVar[str] = "Hulls"
    display_rects_trackbar: ClassVar[str] = "Rectangles"
    frame_trackbar: ClassVar[str] = "Frame"
    ksize_trackbar: ClassVar[str] = "k-Size"
    low_thresh_trackbar: ClassVar[str] = "Low Canny Threshold"
    low_thresh_mult_trackbar: ClassVar[str] = "Low Threshold Scale"
    save_trackbar: ClassVar[str] = "Save"
    window_name: ClassVar[str] = "Adjust LED Detection"
    window_name_image: ClassVar[str] = "Results"

    def __init__(self, frames: _np.ndarray, config: _Config) -> None:
        self.frames = frames
        self.config = config

        self.display_contours = False
        self.display_hulls = False
        self.display_rectangles = False
        self.frame_number = 1
        self.last_result: LedDetectionResults
        self.show = True
        self.size: Optional[Tuple[int, int]] = None
        self.two_windows = False

    def create(self) -> None:
        _cv.namedWindow(self.window_name)
        _cv.createTrackbar(
            self.frame_trackbar,
            self.window_name,
            self.frame_number,
            self.frames.shape[0] - 1,
            self.on_frame_change,
        )
        _cv.createTrackbar(
            self.ksize_trackbar,
            self.window_name,
            self.config.detect.blur.kernel_size.width,
            10,
            self.on_ksize_change,
        )
        _cv.createTrackbar(
            self.aperture_trackbar,
            self.window_name,
            self.config.detect.canny.aperture,
            10,
            self.on_aperture_change,
        )
        _cv.createTrackbar(
            self.low_thresh_trackbar,
            self.window_name,
            self.config.detect.canny.low_threshold,
            100,
            self.on_low_canny_threshold_change,
        )
        _cv.createTrackbar(
            self.low_thresh_mult_trackbar,
            self.window_name,
            self.config.detect.canny.low_threshold_multiplier,
            10,
            self.on_low_threshold_mult_change,
        )
        _cv.createTrackbar(
            self.display_contours_trackbar,
            self.window_name,
            int(self.display_contours),
            int(True),
            self.on_display_contours_change,
        )
        _cv.createTrackbar(
            self.display_hulls_trackbar,
            self.window_name,
            int(self.display_hulls),
            int(True),
            self.on_display_hulls_change,
        )
        _cv.createTrackbar(
            self.display_rects_trackbar,
            self.window_name,
            int(self.display_rectangles),
            int(True),
            self.on_display_rectangles_change,
        )
        _cv.createTrackbar(
            self.save_trackbar,
            self.window_name,
            int(False),
            int(True),
            self.on_save_change,
        )
        self.update_image()

    def on_aperture_change(self, value: int) -> None:
        self.config.detect.canny.aperture = value
        self.update_image()

    def on_display_contours_change(self, value: int) -> None:
        self.display_contours = bool(value)
        self.update_image()

    def on_display_hulls_change(self, value: int) -> None:
        self.display_hulls = bool(value)
        self.update_image()

    def on_display_rectangles_change(self, value: int) -> None:
        self.display_rectangles = bool(value)
        self.update_image()

    def on_frame_change(self, value: int) -> None:
        self.frame_number = value
        self.update_image()

    def on_ksize_change(self, value: int) -> None:
        # kernel size needs to be odd.
        actual = value + 1 if value % 2 == 0 else value
        self.config.detect.blur.kernel_size.width = actual
        self.config.detect.blur.kernel_size.height = actual
        self.update_image()

    def on_low_canny_threshold_change(self, value: int) -> None:
        print(value)
        self.config.detect.canny.low_threshold = value
        print(self.config.detect.canny.low_threshold)
        self.update_image()

    def on_low_threshold_mult_change(self, value: int) -> None:
        self.config.detect.canny.low_threshold_multiplier = value
        self.update_image()

    def on_save_change(self, value: int) -> None:
        if value == 0:
            return
        self.config.save()
        _cv.setTrackbarPos(self.save_trackbar, self.window_name, int(False))

    def update_image(self) -> _np.ndarray:
        result = detect_leds(
            self.frames[self.frame_number],
            (
                self.config.detect.blur.kernel_size.width,
                self.config.detect.blur.kernel_size.height,
            ),
            self.config.detect.canny.aperture,
            self.config.detect.canny.low_threshold,
            self.config.detect.canny.low_threshold_multiplier,
        )
        self.last_result = result
        return self.redraw_image()

    def redraw_image(self) -> _np.ndarray:
        display_frame: _np.ndarray = _cv.cvtColor(
            self.frames[self.frame_number], _cv.COLOR_GRAY2BGR
        )
        if self.display_contours:
            for i in range(len(self.last_result.contours)):
                color = (0, 0, 256)
                _cv.drawContours(display_frame, self.last_result.contours, i, color)
        if self.display_hulls:
            for i in range(len(self.last_result.hulls)):
                color = (0, 256, 0)
                _cv.drawContours(display_frame, self.last_result.hulls, i, color)
        if self.display_rectangles:
            for r in self.last_result.rectangles:
                color = (256, 0, 0)
                _cv.rectangle(
                    display_frame, (r.x, r.y), (r.x + r.width, r.y + r.height), color
                )

        if self.size is not None:
            display_frame = _cv.resize(display_frame, self.size)

        if self.show:
            if self.show and self.two_windows:
                _cv.imshow(self.window_name_image, display_frame)
            else:
                _cv.imshow(self.window_name, display_frame)

        return display_frame

    def wait(self) -> None:
        _cv.waitKey()


if __name__ == "__main__":
    import argparse
    import os.path

    from _util import Config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adjust",
        help="Adjust configuration values in a single window.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--adjust-separate",
        help="Adjust configuration values in two windows.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--display-size",
        help="Resize the image shown in the GUI. Only affects the display size.",
        action="store",
        default=None,
        metavar=("X", "Y"),
        nargs=2,
        type=int,
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Filepath of a valid JSON configuration.",
        metavar="JSON",
        default=os.path.join(os.path.dirname(__file__), "config.json"),
    )
    args = parser.parse_args()

    video = _cv.VideoCapture("target_p04_diff.mp4")
    diff_frames: _np.ndarray = _get_frames(video, grayscale=True)
    fps = video.get(_cv.CAP_PROP_FPS)
    width = int(video.get(_cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(_cv.CAP_PROP_FRAME_HEIGHT))
    video.release()
    del video

    codec = _cv.VideoWriter_fourcc(*"mp4v")
    output = _cv.VideoWriter("target_p05_detect.mp4", codec, fps, (width, height))

    cfg = Config(args.config)
    displaymanager = _DisplayManager(diff_frames, cfg)

    if not args.adjust and not args.adjust_separate:
        print("Detecting LEDs...")
        displaymanager.display_contours = True
        displaymanager.display_hulls = True
        displaymanager.display_rectangles = True
        displaymanager.show = False
        for i in range(displaymanager.frames.shape[0]):
            displaymanager.frame_number = i
            output.write(displaymanager.update_image())
    else:
        if args.display_size is not None:
            displaymanager.size = (args.display_size[0], args.display_size[1])
        displaymanager.two_windows = args.adjust_separate
        displaymanager.create()
        displaymanager.wait()

    output.release()
