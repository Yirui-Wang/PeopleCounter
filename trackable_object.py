import cv2
import numpy as np

OPENCV_TRACKERS = {
    "csrt":         cv2.TrackerCSRT_create,
    "kcf":          cv2.TrackerKCF_create,
    "boosting":     cv2.TrackerBoosting_create,
    "mil":          cv2.TrackerMIL_create,
    "tld":          cv2.TrackerTLD_create,
    "medianflow":   cv2.TrackerMedianFlow_create,
    "mosse":        cv2.TrackerMOSSE_create
}


class TrackableObject:
    def __init__(self, objectID, box, tracker, frame, axis):
        self.objectID = objectID
        self.boxes = [box]
        self.centroids = []
        self.__appendCentroid()
        self.__createTracker(box, tracker, frame)
        self.axis = axis
        self.disappeared = 0
        self.direction = 0
        self.counted = False

    def matchSucceeded(self, box, tracker, frame):
        self.disappeared = 0
        del self.boxes[-1]
        del self.centroids[-1]
        self.boxes.append(box)
        self.__appendCentroid()
        self.__createTracker(box, tracker, frame)

    def trackSucceeded(self, box):
        self.disappeared = 0
        self.boxes.append(box)
        self.__appendCentroid()

    def matchFailed(self):
        self.disappeared += 1

    def __appendCentroid(self):
        (startX, startY, endX, endY) = self.boxes[-1]
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)
        centroid = (cX, cY)
        self.__updateDirection(centroid)
        self.centroids.append(centroid)

    def __createTracker(self, box, tracker, frame):
        (startX, startY, endX, endY) = box.astype("int")
        self.tracker = OPENCV_TRACKERS[tracker]()
        self.tracker.init(frame, (startX, startY, endX - startX, endY - startY))

    def __updateDirection(self, centroid):
        if len(self.centroids) > 10:
            x = [c[self.axis] for c in self.centroids]
            self.direction = centroid[self.axis] - np.mean(x)
