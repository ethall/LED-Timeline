import cv2 as _cv


def extract_valid_frames(source: str, destination: str) -> None:
    # (t,y,x,chan)
    video = _cv.VideoCapture(source)

    codec = _cv.VideoWriter_fourcc(*"mp4v")
    fps = video.get(_cv.CAP_PROP_FPS)
    width = int(video.get(_cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(_cv.CAP_PROP_FRAME_HEIGHT))

    # (t,y,x,chan)
    output = _cv.VideoWriter(destination, codec, fps, (width, height))

    print("Extracting valid frames...\n  valid/total")

    frame_count = int(video.get(_cv.CAP_PROP_FRAME_COUNT))
    for index in range(frame_count):
        success, frame = video.read()
        if not success:
            print("")
            break
        print(f"  {index + 1: 5}/{frame_count}\u000D", end="")
        output.write(frame)
    else:
        print("")

    output.release()
    video.release()


if __name__ == "__main__":
    extract_valid_frames("target.mp4", "target_p01_valid.mp4")
