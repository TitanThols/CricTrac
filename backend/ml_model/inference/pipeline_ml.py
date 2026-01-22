from ultralytics import YOLO
import cv2
import numpy as np
import os
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

        # Probe video properties
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps <= 0:
            fps = 30  # fallback (important)

        # Initialize writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise RuntimeError("❌ VideoWriter failed to open output file")

        print(f"✅ Writing output to: {output_path}")
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


# CLI test
if __name__ == "__main__":
    pipe = MLBatPipeline()
    res = pipe.process_video("sample.mp4", "outputs/ml_test_output.mp4")
    print(res)
