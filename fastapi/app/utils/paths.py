from app.config import UPLOAD_DIR, MODEL_DIR, WEIGHT_DIR
import os

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def get_weights_dir():
    return os.path.join(get_project_root(), "app", MODEL_DIR, WEIGHT_DIR)

def get_weight_path(filename):
    return os.path.join(get_weights_dir(), filename)

def get_upload_dir():
    return os.path.join(get_project_root(), UPLOAD_DIR)