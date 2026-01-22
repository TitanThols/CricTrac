import cv2

class MotionSegmenter:
    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=50,
            varThreshold=12,
            detectShadows=False
        )

    def apply(self, frame):
        fg = self.bg.apply(frame)

        # VERY light cleanup (do NOT kill thin bat)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        return fg
