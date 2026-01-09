# video_io.py
import cv2

def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap

def make_writer(path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, fps, (width, height))

def show_frame(winname, frame, scale=1.0):
    if scale != 1.0:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
    cv2.imshow(winname, frame)
