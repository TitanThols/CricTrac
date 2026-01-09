from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
import uuid
import os
import cv2

from non_ml.pipeline_non_ml import main as run_non_ml

router = APIRouter()

@router.post("/track/non-ml")
async def track_non_ml(file: UploadFile = File(...)):
    # 1) Save uploaded video temporarily
    input_path = f"temp_{uuid.uuid4()}.mp4"
    output_path = f"out_{uuid.uuid4()}.mp4"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 2) Run processing
    run_non_ml(input_path, output_path)

    # 3) Return processed output
    return FileResponse(output_path, media_type="video/mp4")
