from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import os

router = APIRouter()
os.makedirs("uploads", exist_ok=True)

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    saved_files = []
    for file in files:
        contents = await file.read()
        with open(f"uploads/{file.filename}", "wb") as f:
            f.write(contents)
        saved_files.append(file.filename)
    return JSONResponse({"uploaded": saved_files})