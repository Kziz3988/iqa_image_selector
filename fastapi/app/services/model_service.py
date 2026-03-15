from app.models.feature_extractor import ResNetFeatureExtractor
from app.models.iqa import IQAFactory

class ModelService:
    # Model caching
    _feature_extractor = None
    _iqa_models = {}

    @classmethod
    def get_feature_extractor(cls):
        if cls._feature_extractor is None:
            cls._feature_extractor = ResNetFeatureExtractor()
        return cls._feature_extractor

    @classmethod
    def get_iqa_model(cls, model_name: str):
        if model_name not in cls._iqa_models:
            cls._iqa_models[model_name] = IQAFactory.get_model(model_name)
        return cls._iqa_models[model_name]