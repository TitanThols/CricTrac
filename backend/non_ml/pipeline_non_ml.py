import cv2
import sys
import numpy as np

from motion.bg_subtractor import MotionSegmenter   # kept for structure, NOT used
from motion.edge_fusion import fuse_edges_and_motion
from geometry.contour_filter import filter_long_contours
from geometry.min_rect import rect_to_bbox


class BatTracker:
    def __init__(self):
        self.prev_gray = None

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- FRAME DIFFERENCING (KEY FIX) ----
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return {
                "box": None,
                "rect": None,
                "angle": None,
                "mode": "IDLE",
                "combined": None,
                "confidence": 0.0,
                "points": None
            }

        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray.copy()

        # Threshold for FAST motion (bat)
        _, fg = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # VERY light cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

        # ---- EDGE + COLOR FUSION ----
        combined, _ = fuse_edges_and_motion(frame, gray, fg)

        # ---- CONTOURS ----
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = filter_long_contours(contours)

        if not candidates:
            return {
                "box": None,
                "rect": None,
                "angle": None,
                "mode": "IDLE",
                "combined": combined,
                "confidence": 0.0,
                "points": None
            }

        # Pick largest elongated contour
        rect, cnt, area, aspect = candidates[0]
        bbox = rect_to_bbox(rect)

        return {
            "box": bbox,
            "rect": rect,
            "angle": None,
            "mode": "DETECT",
            "combined": combined,
            "confidence": 0.5,
            "points": None
        }


def visualize(frame, result):
    out = frame.copy()

    rect = result.get("rect")
    if rect is not None:
        box = cv2.boxPoints(rect)
        box = box.astype(int)

        cv2.drawContours(out, [box], 0, (0, 255, 0), 2)

        # draw center
        (cx, cy), _, _ = rect
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    return out


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    tracker = BatTracker()

    # ---- FPS FIX (IMPORTANT) ----
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25  # fallback
    delay = int(1000 / fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = tracker.process(frame)
        vis = visualize(frame, result)

        if result.get("combined") is not None:
            cv2.imshow("Combined", result["combined"])

        cv2.imshow("Frame", vis)

        # ---- correct playback speed ----
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline_non_ml.py input.mp4")
        sys.exit(1)

    main(sys.argv[1])
