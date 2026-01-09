from ultralytics import YOLO
import torch

def train_bat_detector():
    # Check GPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {'GPU' if device == 0 else 'CPU'}")
    
    # Load base model
    model = YOLO('yolo11n.pt')
    
    # Train
    results = model.train(
        data='ml_model/training/dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16 if device == 0 else 4,  # Smaller batch for CPU
        patience=20,
        device=device,
        project='ml_model/training/runs',
        name='bat_detection',
        save=True,
        plots=True,
        verbose=True
    )
    
    print("\n✅ Training Complete!")
    print(f"Best model: ml_model/training/runs/bat_detection/weights/best.pt")
    
    return results

if __name__ == "__main__":
    train_bat_detector()