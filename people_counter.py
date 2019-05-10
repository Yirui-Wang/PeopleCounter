import argparse
import time

import cv2
import imutils
import numpy as np
from imutils.video import FPS, FileVideoStream
from people_matcher import PeopleMatcher


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-p",
    "--prototxt",
    default="models/MobileNetSSD_deploy.prototxt",
    help="path to Caffe 'deploy' prototxt file"
)
ap.add_argument(
    "-m",
    "--model",
    default="models/MobileNetSSD_deploy.caffemodel",
    help="path to Caffe pre-trained model"
)
ap.add_argument(
    "-in",
    "--input",
    default="videos/example_01.mp4",
    help="path to optional input video file"
)
ap.add_argument(
    "-a",
    "--axis",
    type=int,
    default=1,
    help="axis of people's position, 0 is horizontal, 1 is vertical"
)
ap.add_argument(
    "-c",
    "--confidence",
    default=0.5,
    help="confidence threshold to filter out weak detections"
)
ap.add_argument(
    "-s",
    "--skip-frames",
    default=15,
    help="number of skip frames between detections"
)
ap.add_argument(
    "-t",
    "--tracker",
    default="kcf",
    help="OpenCV object tracker type"
)
ap.add_argument(
    "-io",
    "--iou",
    type=float,
    default="0.1",
    help="intersection over union"
)
ap.add_argument(
    "-o",
    "--output",
    # default="videos/example_01_count.mp4",
    help="path to optional output video file"
)
args = ap.parse_args()

# initialize a list of class labels MobileNet SSD was trained to detect
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse","motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

# open video file or capture device
if args.input:
    print("[INFO] opening video file...")
    cap = FileVideoStream(args.input).start()
    time.sleep(1.0)
else:
    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(0)

# initialize the frame dimensions
shape = (0, 0)
W = None
H = None

# initialize the total number of frames processed thus far, along with the total
# number of objects that have moved either left(down) or right(up)
totalFrames = 0
totalLeft = 0
totalRight = 0

# initialize the PeopleMatcher object
peopleMatcher = PeopleMatcher()

# initialize the video writer (we'll instantiate later if need be)
writer = None

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
    # capture frame-by-frame
    frame = cap.read()
    if frame is None:
        break

    # resize frame for prediction
    frame = imutils.resize(frame, width=500)

    # if the frame dimensions are empty, set them
    if shape[0] == 0 or shape[1] == 0:
        shape = frame.shape[:2]
        H = shape[0]
        W = shape[1]

    # instantiate the video writer if need be
    if args.output is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        writer = cv2.VideoWriter(args.output, fourcc, 30, (W, H))

    # initialize the current status along with our list of bounding box
    # rectangles returned by either object detector or trackers
    status = "Waiting"

    # update the FPS counter
    fps.update()
    fps.stop()

    # loop over the trackers
    for trackableObject in peopleMatcher.objects:
        # set the status of our system to be 'tracking' rather than 'waiting'
        # or 'detecting'
        status = "Tracking"

        # update the tracker and grab the updated position
        (success, cv2box) = trackableObject.tracker.update(frame)

        (x, y, w, h) = [int(v) for v in cv2box]
        box = [x, y, x + w, y + h]

        # check to see if the tracking was a success
        if success:
            trackableObject.trackSucceeded(box)
        else:
            trackableObject.matchFailed()

    # check to see if we should run a more computationally expensive object
    # detection method to aid our tracker
    if totalFrames % args.skip_frames == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"

        # convert the frame to a blob and pass the blob through the network and
        # obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5, False)
        net.setInput(blob)
        detections = net.forward()

        boxes = []

        # loop over detections
        for i in range(detections.shape[2]):
            # confidence of prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum confidence
            if confidence > args.confidence:
                # extract the index of the class label
                classID = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                if CLASSES[classID] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3: 7] * np.array([W, H, W, H])
                boxes.append(box)

        # match detecting and tracking results
        peopleMatcher.centroid_match(boxes, args.tracker, frame, args.axis)

    for to in peopleMatcher.objects:
        # if the object is in current frame
        if not to.disappeared:
            # count objects
            if not to.counted:
                direction = to.direction
                centroid = to.centroids[-1]
                if direction < 0 and centroid[args.axis] < shape[args.axis] / 2:
                    totalLeft += 1
                    to.counted = True
                elif direction > 0 and centroid[args.axis] > shape[args.axis] / 2:
                    totalRight += 1
                    to.counted = True

            # visualize objects
            box = to.boxes[-1]
            (startX, startY, endX, endY) = [int(v) for v in box]
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0))
            objectID = to.objectID
            centroid = to.centroids[-1]
            cv2.putText(frame, "ID {}".format(objectID), centroid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # construct a tuple of information we will be displaying on the frame
    direction = (("Left", "Right"), ("Up", "Down"))
    info = [
        (direction[args.axis][0], totalLeft),
        (direction[args.axis][1], totalRight),
        ("FPS", "{:.2f}".format(fps.fps())),
        ("Status", status)
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    totalFrames += 1

    cv2.namedWindow("People counter", cv2.WINDOW_NORMAL)
    cv2.imshow("People counter", frame)
    if cv2.waitKey(1) >= 0:  # break with ESC
        break

    if writer is not None:
        writer.write(frame)

if writer is not None:
    writer.release()
