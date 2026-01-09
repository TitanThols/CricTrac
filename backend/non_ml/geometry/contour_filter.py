# contour_filter.py
import cv2
import numpy as np

def filter_long_contours(contours, min_area=1200, aspect_threshold=3.0):
    """
    Filters contours to find elongated objects (like bats).
    Returns:
        candidates: list of (rect, contour, area, aspect_ratio)
    """
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
        (cx, cy), (w, h), ang = rect

        if w == 0 or h == 0:
            continue

        long_side = max(w, h)
        short_side = min(w, h)
        aspect = long_side / (short_side + 1e-6)

        if aspect >= aspect_threshold:
            candidates.append((rect, cnt, area, aspect))

    # Sort by descending area (largest first)
    candidates.sort(key=lambda x: x[2], reverse=True)

    return candidates
