import os
import uuid
import shutil
from app.config import UPLOAD_DIR

def create_task():
    task_id = str(uuid.uuid4())
    task_dir = os.path.join(UPLOAD_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    return task_id, task_dir

def get_task_dir(task_id: str):
    return os.path.join(UPLOAD_DIR, task_id)

def delete_task(task_id: str):
    task_dir = get_task_dir(task_id)
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)