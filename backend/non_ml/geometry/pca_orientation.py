# pca_orientation.py
import numpy as np
import cv2

def contour_pca_angle(cnt):
    pts = np.squeeze(cnt).astype(np.float32)
    if pts.ndim != 2 or pts.shape[0] < 5:
        return None
    mean = pts.mean(axis=0)
    data = pts - mean
    cov = np.cov(data, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # major eigenvector is the last column
    major = eigvecs[:, -1]
    angle = np.degrees(np.arctan2(major[1], major[0]))
    return angle

def min_area_rect(cnt):
    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    return rect, box
