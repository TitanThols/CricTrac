from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import shutil
import os
import subprocess
from pathlib import Path

app = FastAPI(title="CricTrac Bat Tracking API", version="1.0")

BASE_DIR = Path(__file__).resolve().parent

# ==========================================================
#   DIRECTORIES
# ==========================================================
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"

OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Serve output videos
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ==========================================================
#   CORS (React)
# ==========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
#   ROOT
# ==========================================================
@app.get("/")
def root():
    return {
        "message": "CricTrac Backend Running!",
        "endpoints": {
            "ml": "/track/ml",
            "non_ml": "/track/non-ml"
        }
    }

# ==========================================================
#   HEALTH
# ==========================================================
@app.get("/health")
def health():
    return {"status": "healthy"}

# ==========================================================
#   ML PIPELINE (SCRIPT-BASED)
# ==========================================================
@app.post("/track/ml")
async def track_ml(video: UploadFile = File(...)):
    allowed = {".mp4", ".avi", ".mov", ".mkv"}
    ext = Path(video.filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(400, "Invalid video format")

    input_path = UPLOAD_DIR / video.filename
    output_name = f"processed_ml_{video.filename}"
    output_path = OUTPUT_DIR / output_name

    try:
        # Save upload
        with open(input_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        # Run ML pipeline
        cmd = [
            "python3",
            str(BASE_DIR / "ml_model" / "inference" / "pipeline_ml.py"),
            str(input_path),
            str(output_path)
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            raise HTTPException(500, proc.stderr)

        return {
            "status": "success",
            "output_video": f"/outputs/{output_name}"
        }

    finally:
        if input_path.exists():
            os.remove(input_path)

# ==========================================================
#   NON-ML PIPELINE (PLACEHOLDER / SCRIPT)
# ==========================================================
@app.post("/track/non-ml")
async def track_non_ml(video: UploadFile = File(...)):
    allowed = {".mp4", ".avi", ".mov", ".mkv"}
    ext = Path(video.filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(400, "Invalid video format")

    input_path = UPLOAD_DIR / video.filename
    output_name = f"processed_non_ml_{video.filename}"
    output_path = OUTPUT_DIR / output_name

    try:
        with open(input_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        cmd = [
            "python3",
            str(BASE_DIR / "non_ml" / "pipeline_non_ml.py"),
            str(input_path),
            str(output_path)
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            raise HTTPException(500, proc.stderr)

        return {
            "status": "success",
            "output_video": f"/outputs/{output_name}"
        }

    finally:
        if input_path.exists():
            os.remove(input_path)

# ==========================================================
#   RUN SERVER
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
