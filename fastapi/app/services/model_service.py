from app.models.feature_extractor import ExtractorFactory
from app.models.clustering import ClustererFactory
from app.models.iqa import IQAFactory

class ModelService:
    # Model caching
    _models = {}

    @staticmethod
    def initialize_models(
        models = {
            "ResNet": "extractor",
            "Agglomerative": "clusterer",
            "HDBSCAN": "clusterer",
            "DBCNN": "iqa",
            "MANIQA": "iqa",
            "ARNIQA": "iqa",
            "Selector": "iqa"
        }):
        for name, type in models.items():
            ModelService._models[name] = ModelService.get_model(type, name)

    @staticmethod
    def get_model(model_type: str, model_name: str):
        if model_name not in ModelService._models:
            if model_type == "extractor":
                ModelService._models[model_name] = ExtractorFactory.get_model(model_name)
            elif model_type == "clusterer":
                ModelService._models[model_name] = ClustererFactory.get_model(model_name)
            elif model_type == "iqa":
                ModelService._models[model_name] = IQAFactory.get_model(model_name)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        return ModelService._models[model_name]
