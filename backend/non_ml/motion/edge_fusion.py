import cv2
import numpy as np

def fuse_edges_and_motion(frame, gray, fg):
    """
    frame : BGR frame
    gray  : grayscale image
    fg    : foreground mask from bg subtractor
    Returns:
        combined_mask : fused edge-motion-HSV mask
        edges         : raw Canny edges
    """

    # --- 1) Blur to reduce fine net edges ---
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 2) Strong Canny keeps only real edges (bat, player) ---
    edges = cv2.Canny(blur, 120, 250)

    # --- 3) Combine Canny edges with foreground motion ---
    combined = cv2.bitwise_and(edges, fg)

    # --- 4) HSV bat color boost ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # light-brown bat tone range (tuned for typical wooden bat)
    lower_bat = np.array([5, 30, 40])
    upper_bat = np.array([25, 255, 255])

    mask_bat = cv2.inRange(hsv, lower_bat, upper_bat)

    # merge bat mask with edge-motion mask
    combined = cv2.bitwise_or(combined, mask_bat)

    # --- 5) Morphological closing to group bat edges into a big blob ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    return combined, edges
