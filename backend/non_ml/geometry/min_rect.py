# min_rect.py
import cv2
import numpy as np

def compute_min_rect(cnt):
    """
    Wrapper around cv2.minAreaRect to return:
        - rect: ((cx,cy),(w,h),angle)
        - box: 4 corner points as int array
    """
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    return rect, box


def rect_to_bbox(rect):
    """
    Convert cv2.minAreaRect to simpler (x,y,w,h) bounding box.
    """
    (cx, cy), (w, h), angle = rect
    x = int(cx - w/2)
    y = int(cy - h/2)
    return x, y, int(w), int(h)
