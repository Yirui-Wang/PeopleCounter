import cv2
import numpy as np
from scipy.spatial import distance as dist
from trackable_object import TrackableObject


class PeopleMatcher:
    def __init__(self):
        self.nextObjectID = 1
        self.objects = []
        self.maxDisappeared = 50
        self.maxDistance = 200
        self.minIoU = 0.2

    def register(self, box, tracker, frame):
        to = TrackableObject(self.nextObjectID, box, tracker, frame)
        self.nextObjectID += 1
        self.objects.append(to)

    def deregister(self, index):
        del self.objects[index]

    def centroid_match(self, boxes, tracker, frame):
        if len(boxes) == 0:
            for (i, to) in enumerate(self.objects):
                to.matchFailed()
                if to.disappeared > self.maxDisappeared:
                    self.deregister(i)
            return

        inputCentroids = np.zeros((len(boxes), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(boxes):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(boxes[i], tracker, frame)

        objectCentroids = []
        for to in self.objects:
            objectCentroids.append(to.centroids[-1])

        D = dist.cdist(np.array(objectCentroids), inputCentroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue

            if D[row, col] > self.maxDistance:
                continue

            self.objects[row].matchSucceeded(boxes[col], tracker, frame)

            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)

        if D.shape[0] >= D.shape[1]:
            for row in unusedRows:
                to = self.objects[row]
                to.matchFailed()

                if to.disappeared > self.maxDisappeared:
                    self.deregister(row)
        else:
            for col in unusedCols:
                self.register(boxes[col], tracker, frame)

    @staticmethod
    def __bb_intersection_over_union(box1, box2):
        # determine the (x, y)-coordinates of the intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the area of intersection rectangle
        inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # compute the area of both the prediction and ground-truth rectangles
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # compute the intersection over union by taking the intersection area and
        # dividing it by the sum of prediction + ground-truth areas - the
        # intersection area
        iou = inter_area / float(box1_area + box2_area - inter_area)

        # return the intersection over union value
        return iou

    def iou_match(self, boxes, tracker, frame):
        if len(boxes) == 0:
            for (i, to) in enumerate(self.objects):
                to.matchFailed()
                if to.disappeared > self.maxDisappeared:
                    self.deregister(i)
            return

        if len(self.objects) == 0:
            for i in range(len(boxes)):
                self.register(boxes[i], tracker, frame)

        IoU = np.zeros((len(self.objects), len(boxes)), dtype="int")
        for i in range(len(self.objects)):
            for j in range(len(boxes)):
                IoU[i, j] = self.__bb_intersection_over_union(
                    self.objects[i].boxes[-1], boxes[j])

        rows = IoU.max(axis=1).argsort()
        cols = IoU.argmax(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue

            if IoU[row, col] > self.maxDistance:
                continue

            self.objects[row].matchSucceeded(boxes[col], tracker, frame)

            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(0, IoU.shape[0])).difference(usedRows)
        unusedCols = set(range(0, IoU.shape[1])).difference(usedCols)

        if IoU.shape[0] >= IoU.shape[1]:
            for row in unusedRows:
                to = self.objects[row]
                to.matchFailed()

                if to.disappeared > self.maxDisappeared:
                    self.deregister(row)
        else:
            for col in unusedCols:
                self.register(boxes[col], tracker, frame)
