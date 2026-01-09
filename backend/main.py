from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import shutil
import os
import subprocess
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

# ==========================================================
#   ENABLE STATIC FILES (so frontend can load output videos)
# ==========================================================
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# FastAPI now serves files from /outputs at:
#   http://localhost:8000/outputs/<filename>
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ==========================================================
#   CORS for React frontend
# ==========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # change to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "CricTrac Backend Running!"}


# ==========================================================
#   NON-ML PIPELINE ENDPOINT
# ==========================================================
@app.post("/track/non-ml")
async def track_non_ml(video: UploadFile = File(...)):
    # Store uploaded video temporarily
    input_path = BASE_DIR / f"temp_{video.filename}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Output location inside /outputs so frontend can fetch it
    output_path = OUTPUT_DIR / f"processed_{video.filename}"

    # Build command to run pipeline script
    cmd = [
        "python3",
        str(BASE_DIR / "non_ml" / "pipeline_non_ml.py"),
        str(input_path),
        str(output_path)
    ]

    # Run the script
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        return {
            "status": "error",
            "stderr": proc.stderr,
            "stdout": proc.stdout
        }

    return {
        "status": "success",
        "output_video": f"/outputs/processed_{video.filename}"
    }


# ==========================================================
#   PLACEHOLDER ML PIPELINE (we implement later)
# ==========================================================
@app.post("/track/ml")
async def track_ml(video: UploadFile = File(...)):
    temp_path = OUTPUT_DIR / f"ml_{video.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    return {
        "status": "ML pipeline not implemented yet",
        "saved_at": f"/outputs/ml_{video.filename}"
    }


# ==========================================================
#   RUN UVICORN (if started manually)
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=8000,
                reload=True)
