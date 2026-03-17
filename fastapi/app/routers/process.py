from fastapi import APIRouter
import asyncio
from app.utils.task import get_task_dir, delete_task
from app.utils.progress import progress_manager
from app.services.model_service import ModelService
from app.models.clustering import CosineClusterer, HDBSCANClusterer
from app.config import PROCESS_ROUTE
import os

router = APIRouter()
@router.get(f"{PROCESS_ROUTE}/{{task_id}}")

async def process_images(task_id: str, iqa_model: str = "ARNIQA"):
    task_dir = get_task_dir(task_id)
    if not os.path.exists(task_dir):
        return {"error": "Task not found"}

    files = os.listdir(task_dir)
    file_paths = [os.path.join(task_dir, f) for f in files]
    if not files:
        return {"error": "No images"}

    try:
        # Extract features
        await progress_manager.send(task_id, {
            "stage": "feature",
            "message": "提取特征..."
        })
        extractor = ModelService.get_feature_extractor()
        features = [extractor.extract(f) for f in file_paths]

        # Cluster features
        await progress_manager.send(task_id, {
            "stage": "cluster",
            "message": "特征聚类..."
        })
        clusterer = CosineClusterer() if len(features) < 10 else HDBSCANClusterer()
        labels = clusterer.cluster(features)

        # Score with IQA
        await progress_manager.send(task_id, {
            "stage": "iqa",
            "message": "图像质量评分..."
        })
        iqa = ModelService.get_iqa_model(iqa_model)
        scores = [iqa.predict(f) for f in file_paths]

        # Select all best images
        cluster_results = {}
        for file, label, score in zip(files, labels, scores):
            label = int(label)
            if label not in cluster_results or score > cluster_results[label]["score"]:
                cluster_results[label] = {
                    "file": os.path.basename(file),
                    "score": score
                }

        # Return the result
        await progress_manager.send(task_id, {
            "stage": "done",
            "message": "处理完毕"
        })
        result = {
            "task_id": task_id,
            "all_files": files,
            "labels": [int(l) for l in labels.tolist()],
            "scores": [float(s) for s in scores],
            "best_in_cluster": {str(k): os.path.basename(v["file"]) for k, v in cluster_results.items()}
        }
        return result

    finally:
        delete_task(task_id)