# pipeline_non_ml.py
# Full corrected non-ML pipeline orchestrator
# Place this file at backend/non_ml/pipeline_non_ml.py
# Run from backend/: python3 non_ml/pipeline_non_ml.py input.mp4 [output.mp4]

import sys
from pathlib import Path
# ensure backend folder is on sys.path so package imports work when running from backend
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from collections import deque

from non_ml.motion.bg_subtractor import MotionSegmenter
from non_ml.motion.edge_fusion import fuse_edges_and_motion
from non_ml.geometry.contour_filter import filter_long_contours
from non_ml.geometry.pca_orientation import contour_pca_angle
from non_ml.geometry.min_rect import rect_to_bbox
from non_ml.tracking.kalman import KalmanCentroid
from non_ml.tracking.optical_flow import PatchOpticalFlow
from utils.video_io import open_video, make_writer, show_frame

print("Pipeline started")

class NonMLPipeline:
    def __init__(self, min_area=4000, aspect_thr=2.2, max_lost=20):
        self.motion = MotionSegmenter()
        self.kalman = KalmanCentroid()
        self.of = PatchOpticalFlow()
        self.trace = deque(maxlen=128)
        self.min_area = min_area
        self.aspect_thr = aspect_thr
        self.lost = 0
        self.max_lost = max_lost

    def process(self, frame):
        # convert to gray for edges & flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # motion mask
        fg = self.motion.apply(frame)

        # edges fused with motion
        combined, edges = fuse_edges_and_motion(frame, gray, fg)

        # find contours on combined mask
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = filter_long_contours(contours, self.min_area, self.aspect_thr)

        rect = None
        cnt = None
        detected = False

        # If Kalman is initialized, predict and associate with nearest candidate
        if self.kalman.initialized and candidates:
            px, py = self.kalman.predict()
            best_dist = float("inf")
            best = None

            for cand in candidates:
                # cand is (rect, contour, area, aspect)
                rect_cand, cnt_cand, area_cand, aspect_cand = cand
                (cx_cand, cy_cand), _, _ = rect_cand
                d = np.hypot(cx_cand - px, cy_cand - py)
                if d < best_dist:
                    best_dist = d
                    best = cand

            if best and best_dist < max(frame.shape[:2]) * 0.25:
                rect, cnt, _, _ = best
                detected = True

        # If nothing matched or kalman not initialized, pick largest candidate if available
        if rect is None and candidates:
            rect, cnt, _, _ = candidates[0]
            detected = True

        if detected and rect is not None:
            (cx, cy), (w, h), ang = rect
            # robust orientation via PCA if possible
            angle = contour_pca_angle(cnt) or ang

            # initialize or update kalman
            if not self.kalman.initialized:
                self.kalman.init(cx, cy)
            else:
                self.kalman.update(cx, cy)

            # init optical flow patch if not already
            if self.of.prev_pts is None:
                self.of.init(gray, rect)

            # tracing and bookkeeping
            self.trace.append((int(cx), int(cy)))
            self.lost = 0

            bbox = rect_to_bbox(rect)
            return {"box": bbox, "angle": angle, "mode": "DETECT", "combined": combined}

        # Optical flow fallback: get median translation and use as pseudo-measurement
        ofres = self.of.step(gray)
        if ofres is not None and self.kalman.initialized:
            dx, dy, pts = ofres
            # current state post may not exist if kalman not init, but we checked above
            px = float(self.kalman.kf.statePost[0,0]) + float(dx)
            py = float(self.kalman.kf.statePost[1,0]) + float(dy)
            self.kalman.update(px, py)
            self.trace.append((int(px), int(py)))
            self.lost = 0
            return {"box": (int(px-20), int(py-20), 40, 40), "angle": None, "mode": "OF", "combined": combined}

        # Nothing found this frame
        if self.kalman.initialized:
            self.lost += 1
            px, py = self.kalman.predict()
            self.trace.append((int(px), int(py)))
            # reset if lost for long
            if self.lost > self.max_lost:
                self.kalman.initialized = False
                self.of.prev_pts = None
                self.of.prev_gray = None
                self.trace.clear()
                self.lost = 0
            return {"box": None, "angle": None, "mode": "LOST", "combined": combined}
        else:
            return {"box": None, "angle": None, "mode": "SEARCH", "combined": combined}


def visualize(frame, result, trace):
    out = frame.copy()
    if result.get("box"):
        x, y, w, h = result["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, result["mode"], (max(5, x), max(20, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # draw trace
    if trace:
        pts = list(trace)
        for i in range(1, len(pts)):
            cv2.line(out, pts[i - 1], pts[i], (0, 200, 255), 2)
    return out


def main(video_path, out_path=None):
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = make_writer(out_path, fps, w, h) if out_path else None
    pipe = NonMLPipeline()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipe.process(frame)
        vis = visualize(frame, result, pipe.trace)

        # show main tracker and combined mask (resized)
        show_frame("Tracker", vis)
        if result.get("combined") is not None:
            try:
                cm = cv2.resize(result["combined"], (max(1, w // 3), max(1, h // 3)))
                show_frame("Mask/Combined", cm)
            except Exception:
                # in case resize fails, just show original
                show_frame("Mask/Combined", result["combined"])

        if writer:
            writer.write(vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 pipeline_non_ml.py input.mp4 [output.mp4]")
        sys.exit(1)
    p = sys.argv[1]
    o = sys.argv[2] if len(sys.argv) > 2 else None
    main(p, o)
