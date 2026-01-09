from ml_model.inference.pipeline_ml import MLBatPipeline
import os

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

# Initialize and run
pipeline = MLBatPipeline()
results = pipeline.process_video('sample.mp4', 'outputs/ml_output.mp4')

print("\n" + "="*50)
print("ML DETECTION RESULTS")
print("="*50)
print(f"Total frames: {results['total_frames']}")
print(f"Total detections: {results['total_detections']}")
print(f"Unique bats tracked: {results['unique_bats']}")