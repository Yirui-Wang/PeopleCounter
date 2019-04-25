import argparse
import cv2
import imutils
import numpy as np
from imutils.video import FPS
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
    default="videos/test1.mp4",
    help="path to optional input video file"
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
    default=23,
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
    default="videos/test1_result_fault.mp4",
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
    cap = cv2.VideoCapture(args.input)
else:
    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(0)

# initialize the frame dimensions
W = None
H = None

# initialize the total number of frames processed thus far
totalFrames = 0

# start the frames per second throughput estimator
fps = FPS().start()

peopleMatcher = PeopleMatcher()

writer = None

# loop over frames from the video stream
while True:
    # capture frame-by-frame
    ret, frame = cap.read()

    if frame is None:
        break

    # resize frame for prediction
    frame = imutils.resize(frame, width=500)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    if args.output is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        writer = cv2.VideoWriter(args.output, fourcc, 30, (W, H))

    # initialize the current status along with our list of bounding box
    # rectangles returned by either object detector or trackers
    status = "Waiting"

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
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                boxes.append(box)

                # # draw location of object
                # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0))

        # their is no trackable
        # if trackableObjects.len == 0:
        #     for box in boxes:
        #         trackableObjects.add(box, args.tracker, frame)
        # else:
        # intersection_over_union_match(boxes, args.tracker, frame)
        peopleMatcher.centroid_match(boxes, args.tracker, frame)

    # otherwise, we should utilize our object *trackers* rather than object
    # *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for trackableObject in peopleMatcher.objects:
            # set the status of our system to be 'tracking' rather than
            # 'waiting' or 'detecting'
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
                # (x, y, w, h) = [int(v) for v in box]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))

            # add the bounding box coordinates to the rectangles list
            # rects.append((startX, startY, endX, endY))

    for trackableObject in peopleMatcher.objects:
        if not trackableObject.disappeared:
            box = trackableObject.boxes[-1]
            (startX, startY, endX, endY) = [int(v) for v in box]
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0))
            objectID = trackableObject.objectID
            centroid = trackableObject.centroids[-1]
            cv2.putText(frame, "ID {}".format(objectID), centroid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [("Status", status)]

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

