# LED Timeline

> Track LED on and off states with OpenCV.

A small personal project to see what's possible with Python + OpenCV.

## &#9888; WARNING &#9888;

This code is awful. The whole thing is a hack. Use at your discretion.

## Requirements

1. Python 3.7+
1. OpenCV<br/>
I built my libraries from source (specifically, commit [c722625f28](https://github.com/opencv/opencv/tree/c722625f280258f5c865002899bf0dc2ebff1b2b)). Pre-built binaries *should* work, but your milage may vary.

## Steps

1. Extract valid frames.<br/>
Ran into an issue where trimming the video with the Android camera app resulted in a frame count larger than the number of readable frames. This is mitigated by extracting all readable frames from the given video.
1. Denoising via the non-local means algorithm.<br/>
The goal here is to smooth digital sensor noise in order to reduce its impact on changes in pixel value over time. This is extremely CPU intensive and will take a while to complete. While offloading this process to the GPU is an option, it makes the code (even) less portable since the CUDA SDK is required.
1. Convert the video to grayscale.
1. Calculate the change in pixel intensity over time.<br/>
This causes each LED appear as though it only turns on for a single frame.
1. Using Canny edge detection and contour tracing, search each frame and attempt to resolve a bounding rectangle. Once all bounding rectangles have been identified, the grayscale video is replayed and each area is analyzed. If the average pixel value of an area defined by a bounding rectange reaches a certain threshold, then that LED is considered "on".

## License

> MIT License
>
> Copyright (c) 2020 Evan Hall

See the [LICENSE](./LICENSE) file for more details.
