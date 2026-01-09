# optical_flow.py
import cv2
import numpy as np

class PatchOpticalFlow:
    def __init__(self, max_corners=150, quality=0.01, min_dist=7):
        self.lk_params = dict(winSize=(15,15), maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray = None
        self.prev_pts = None
        self.max_corners = max_corners
        self.quality = quality
        self.min_dist = min_dist

    def init(self, gray, rect, expand=1.2):
        (cx, cy), (w, h), angle = rect
        size = int(max(w, h) * expand)
        x1 = int(max(0, cx - size/2)); y1 = int(max(0, cy - size/2))
        x2 = int(min(gray.shape[1], cx + size/2)); y2 = int(min(gray.shape[0], cy + size/2))
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            self.prev_pts = None
            return
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=self.max_corners,
                                      qualityLevel=self.quality, minDistance=self.min_dist)
        if pts is None:
            self.prev_pts = None
            return
        pts[:,0,0] += x1
        pts[:,0,1] += y1
        self.prev_pts = pts
        self.prev_gray = gray.copy()

    def step(self, gray):
        if self.prev_pts is None or self.prev_gray is None:
            return None
        nextPts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self.lk_params)
        if nextPts is None:
            self.prev_pts = None
            return None
        status = status.reshape(-1)
        good_prev = self.prev_pts[status==1]
        good_next = nextPts[status==1]
        if len(good_next) < 3:
            self.prev_pts = None
            return None
        dx = np.median(good_next[:,0,0] - good_prev[:,0,0])
        dy = np.median(good_next[:,0,1] - good_prev[:,0,1])
        self.prev_pts = good_next.reshape(-1,1,2)
        self.prev_gray = gray.copy()
        return dx, dy, good_next
