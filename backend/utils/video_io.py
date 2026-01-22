# video_io.py
# Place at: backend/utils/video_io.py

import cv2
import sys

def open_video(path):
    """Open video file and return capture object"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {path}")
        sys.exit(1)
    return cap

def make_writer(path, fps, width, height):
    """Create video writer"""
    if path is None:
        return None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Warning: Cannot create output video {path}")
        return None
    return writer

def show_frame(window_name, frame):
    """Display frame in named window"""
    cv2.imshow(window_name, frame)