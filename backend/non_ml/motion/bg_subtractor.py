import cv2
import numpy as np

class MotionSegmenter:
    def __init__(self, history=200, varThreshold=25, detectShadows=False, morph_kernel=(5,5)):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=history,
                                                     varThreshold=varThreshold,
                                                     detectShadows=detectShadows)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)

    def apply(self, frame):
        fg = self.bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, self.kernel, iterations=1)
        # optional: threshold to make mask binary
        _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
        return fg
