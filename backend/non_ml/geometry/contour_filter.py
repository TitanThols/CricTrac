import cv2

def filter_long_contours(contours):
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), _ = rect

        if w == 0 or h == 0:
            continue

        aspect = max(w, h) / (min(w, h) + 1e-6)

        # VERY RELAXED
        if aspect > 2.0:
            candidates.append((rect, cnt, area, aspect))

    # biggest first
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates
