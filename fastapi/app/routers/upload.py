from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
from app.utils.task import create_task
from app.utils.progress import progress_manager
from app.config import UPLOAD_DIR, UPLOAD_ROUTE
import os

router = APIRouter()
os.makedirs(UPLOAD_DIR, exist_ok=True)
@router.post(UPLOAD_ROUTE)

async def upload_files(files: List[UploadFile] = File(...)):
    task_id, task_dir = create_task()
    saved_files = []
    await progress_manager.send(task_id, {
        "stage": "ready",
        "message": "上传图像..."
    })

    for file in files:
        path = os.path.join(task_dir, file.filename)
        contents = await file.read()
        with open(path, "wb") as f:
            f.write(contents)
        saved_files.append(file.filename)

    return JSONResponse({
        "task_id": task_id,
        "uploaded": saved_files
    })