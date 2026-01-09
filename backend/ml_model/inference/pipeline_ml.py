from ultralytics import YOLO
import cv2
import numpy as np

class MLBatPipeline:
    def __init__(self, model_path='ml_model/model_weights/best.pt'):
        """Initialize ML-based bat detection pipeline"""
        self.model = YOLO(model_path)
        
    def process_video(self, video_path, output_path, conf_threshold=0.4):
        """
        Process video and track bats using ML model
        
        Args:
            video_path: Input video path
            output_path: Output video path
            conf_threshold: Confidence threshold for detections
            
        Returns:
            dict: Processing results with detections and metrics
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_idx = 0
        
        print(f"Processing {total_frames} frames...")
        
        # Use YOLO tracking
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
                
                # Get track IDs if available
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = [-1] * len(boxes)
                
                # Draw detections
                for box, conf, track_id in zip(boxes, confs, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"Bat ID:{track_id} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Store detection
                    all_detections.append({
                        'frame': frame_idx,
                        'track_id': int(track_id),
                        'bbox': [x1, y1, x2-x1, y2-y1],
                        'confidence': float(conf)
                    })
            
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        # Calculate metrics
        unique_tracks = len(set([d['track_id'] for d in all_detections if d['track_id'] != -1]))
        
        results_dict = {
            'total_frames': frame_idx,
            'total_detections': len(all_detections),
            'unique_bats': unique_tracks,
            'detections': all_detections,
            'output_video': output_path,
            'fps': fps
        }
        
        print(f"\n✅ Processing complete!")
        print(f"Total detections: {len(all_detections)}")
        print(f"Unique bats tracked: {unique_tracks}")
        
        return results_dict

# Test function
if __name__ == "__main__":
    pipeline = MLBatPipeline()
    results = pipeline.process_video('sample.mp4', 'outputs/ml_output.mp4')
    print(results)