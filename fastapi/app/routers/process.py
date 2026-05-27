from fastapi import APIRouter
import numpy as np
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
        extractor = ModelService.get_model("extractor", "ResNet")
        features = {}
        n_files = len(file_paths)
        for idx, fpath in enumerate(file_paths, 1):
            filename_with_ext = os.path.basename(fpath)
            filename = os.path.splitext(filename_with_ext)[0]
            features[filename] = {
                'path': fpath,
                'vector': extractor.extract(fpath)
            }
            await progress_manager.send(task_id, {
                "stage": "feature",
                "message": f"提取特征...({idx}/{n_files})",
                "progress": idx / n_files
            })

        # Cluster features
        await progress_manager.send(task_id, {
            "stage": "cluster",
            "message": "特征聚类..."
        })
        feature_vectors = [f['vector'] for f in features.values()]
        clusterer = ModelService.get_model("clusterer", "Agglomerative") if n_files < 50 else ModelService.get_model("clusterer", "HDBSCAN")
        labels = clusterer.cluster(feature_vectors)

        # Score with IQA
        scores = []
        iqa_models = []
        filenames = list(features.keys())

        iqa = ModelService.get_model("iqa", iqa_model)
        if iqa_model == "Selector" and iqa.iqa is None:
            iqa.iqa = [ModelService.get_model("iqa", name) for name in iqa.iqa_names]
        
        if iqa_model == "Selector":
            all_features = np.array([f['vector'] for f in features.values()])
            feat_mean = all_features.mean(axis=0)
            feat_std = all_features.std(axis=0) + 1e-8
            for filename in filenames:
                features[filename]['vector'] = (features[filename]['vector'] - feat_mean) / feat_std
            
            for idx, filename in enumerate(filenames, 1):
                score, model_name = iqa.predict(
                    features[filename]['path'],
                    features[filename]['vector']
                )
                scores.append(score)
                iqa_models.append(model_name)
                await progress_manager.send(task_id, {
                    "stage": "iqa",
                    "message": f"图像质量评分...({idx}/{n_files})",
                    "progress": idx / n_files
                })
        else:
            for idx, filename in enumerate(filenames, 1):
                fpath = features[filename]['path']
                score = iqa.predict(fpath)
                scores.append(score)
                iqa_models.append(iqa_model)
                await progress_manager.send(task_id, {
                    "stage": "iqa",
                    "message": f"图像质量评分...({idx}/{n_files})",
                    "progress": idx / n_files
                })

        # Select all best images
        file_data = {}
        for filename, label, score, model in zip(filenames, labels, scores, iqa_models):
            label = int(label)
            if label not in file_data:
                file_data[label] = []
            file_data[label].append({
                "uuid": filename,
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