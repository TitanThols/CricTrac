import cv2
import numpy as np

def fuse_edges_and_motion(frame, gray, fg):
    # Edges
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 60, 160)

    # Motion + edges
    motion_edges = cv2.bitwise_and(edges, fg)

    # HSV bat color (KEEP IT SIMPLE)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_bat = np.array([5, 30, 60])
    upper_bat = np.array([35, 255, 255])
    mask_color = cv2.inRange(hsv, lower_bat, upper_bat)

    # Bat must be MOVING
    moving_color = cv2.bitwise_and(mask_color, fg)

    # FINAL COMBINATION
    combined = cv2.bitwise_or(motion_edges, moving_color)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,8))
    # Thin out thick regions (kills body, keeps bat)
    thin_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.erode(combined, thin_kernel, iterations=2)

    # Reconnect thin structures
    combined = cv2.dilate(combined, thin_kernel, iterations=1)


    return combined, edges
