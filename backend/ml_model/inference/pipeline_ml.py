from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
import subprocess
from pathlib import Path

class MLBatPipeline:
    def __init__(self, model_path='ml_model/model_weights/best.pt'):
        self.model = YOLO(model_path)

    def process_video(self, video_path, output_path, conf_threshold=0.4):
        video_path = str(video_path)
        output_path = str(output_path)

        # Ensure output directory exists
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use temporary output with different codec
        temp_output = output_path.replace('.mp4', '_temp.avi')

        # Probe video properties
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps <= 0:
            fps = 30

        # Write to AVI first (more compatible)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        if not out.isOpened():
            raise RuntimeError("❌ VideoWriter failed to open output file")

        print(f"✅ Writing temporary output to: {temp_output}")
        print(f"Video: {width}x{height} @ {fps} FPS")

        all_detections = []
        frame_idx = 0

        results = self.model.track(
            source=video_path,
            conf=conf_threshold,
            iou=0.5,
            tracker='bytetrack.yaml',
            stream=True,
            verbose=False
        )

        for result in results:
            frame = result.orig_img.copy()

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = [-1] * len(boxes)

                for box, conf, tid in zip(boxes, confs, track_ids):
                    x1, y1, x2, y2 = map(int, box)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Bat {tid} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                    all_detections.append({
                        "frame": frame_idx,
                        "track_id": int(tid),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "confidence": float(conf)
                    })

            out.write(frame)
            frame_idx += 1

        out.release()

        print("🔄 Converting to browser-compatible MP4...")
        
        # Convert to MP4 with ffmpeg
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_output,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                output_path
            ], check=True, capture_output=True, text=True)
            
            # Remove temp file
            os.remove(temp_output)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg conversion failed: {e.stderr}")
            raise RuntimeError("FFmpeg conversion failed")

        unique_tracks = len(
            set(d["track_id"] for d in all_detections if d["track_id"] != -1)
        )

        print("✅ ML processing finished")

        return {
            "total_frames": frame_idx,
            "total_detections": len(all_detections),
            "unique_bats": unique_tracks,
            "output_video": output_path,
            "fps": fps
        }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pipeline_ml.py <input_video> <output_video>")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    
    pipe = MLBatPipeline()
    res = pipe.process_video(input_video, output_video)
    print(res)