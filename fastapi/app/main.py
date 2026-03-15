from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routers.upload import router as upload_router
from app.routers.process import router as process_router
from app.routers.ws import router as ws_router
from app.config import UPLOAD_DIR

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(process_router)
app.include_router(ws_router)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name=UPLOAD_DIR)
@app.get("/")

async def root():
    return {"message": "API is running"}