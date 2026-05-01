from fastapi import APIRouter
from app.utils.task import get_task_dir, delete_task
from app.utils.progress import progress_manager
from app.services.model_service import ModelService
from app.models.clustering import AgglomerativeClusterer, HDBSCANClusterer
from app.config import PROCESS_ROUTE
import os

router = APIRouter()
@router.get(f"{PROCESS_ROUTE}/{{task_id}}")

async def process_images(task_id: str, iqa_model: str = "Selector"):
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
        extractor = ModelService.get_model("extractor", "ResNet")
        features = {f: extractor.extract(f) for f in file_paths}
        n_samples = len(features)

        # Cluster features
        await progress_manager.send(task_id, {
            "stage": "cluster",
            "message": "特征聚类..."
        })
        clusterer = ModelService.get_model("clusterer", "Agglomerative") if n_samples < 50 else ModelService.get_model("clusterer", "HDBSCAN")
        labels = clusterer.cluster(list(features.values()))

        # Score with IQA
        await progress_manager.send(task_id, {
            "stage": "iqa",
            "message": "图像质量评分..."
        })
        iqa = ModelService.get_model("iqa", iqa_model)
        if iqa_model == "Selector":
            scores = []
            iqa_models = []
            if iqa.iqa == None:
                iqa.iqa = [ModelService.get_model("iqa", name) for name in iqa.iqa_names]
            for k, v in features.items():
                score, iqa_name = iqa.predict(k, v)
                scores.append(score)
                iqa_models.append(iqa_name)
        else:
            scores = [iqa.predict(f) for f in file_paths]
            iqa_models = [iqa_model for f in file_paths]

        # Select all best images 
        file_data = {}
        for file, label, score, model in zip(files, labels, scores, iqa_models):
            label = int(label)
            if label not in file_data:
                file_data[label] = []
            file_data[label].append({
                "file": os.path.basename(file),
                "score": float(score),
                "model": model
            })
        sorted_file_data = []
        for label in sorted(file_data.keys()):
            sorted_data = sorted(file_data[label], key=lambda x: x['score'], reverse=True)
            sorted_file_data.append({
                "cluster": label,
                "best_image": sorted_data[0],
                "other_images": sorted_data[1:]
            })

        # Return the result
        await progress_manager.send(task_id, {
            "stage": "done",
            "message": "处理完毕"
        })
        result = {
            "task_id": task_id,
            "file_data": sorted_file_data
        }
        return result

    finally:
        delete_task(task_id)