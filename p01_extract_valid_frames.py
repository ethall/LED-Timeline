import cv2 as cv


# (t,y,x,chan)
video = cv.VideoCapture("target.mp4")

codec = cv.VideoWriter_fourcc(*"mp4v")
fps = int(video.get(cv.CAP_PROP_FPS))
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

# (t,y,x,chan)
output = cv.VideoWriter("target_p01_valid.mp4", codec, fps, (width, height))

print("Extracting valid frames...\n  valid/total")

frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
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
