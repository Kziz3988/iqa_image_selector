import torch
import torch.nn as nn
from typing import Tuple
from .models.resnet import ResNet
from app.utils.paths import get_weight_path

available_datasets_ranges = {
    "live": (1, 100),
    "csiq": (0, 1),
    "tid2013": (0, 9),
    "kadid10k": (1, 5),
    "flive": (1, 100),
    "spaq": (1, 100),
    "clive": (1, 100),
    "koniq10k": (1, 100),
}

available_datasets_mos_types = {
    "live": "dmos",
    "csiq": "dmos",
    "tid2013": "mos",
    "kadid10k": "mos",
    "flive": "mos",
    "spaq": "mos",
    "clive": "mos",
    "koniq10k": "mos",
}

class ARNIQA(nn.Module):
    def __init__(self):
        super(ARNIQA, self).__init__()
        model_weight_path = get_weight_path('arniqa.pth')
        regressor_weight_path = get_weight_path('regressor_kadid10k.pth')
        self.regressor_dataset = 'kadid10k'
        self.encoder = ResNet(embedding_dim=128, pretrained=False, use_norm=True)
        self.encoder.load_state_dict(torch.load(model_weight_path, map_location="cpu", weights_only=False))
        self.encoder.eval()
        self.regressor: nn.Module = torch.load(regressor_weight_path, map_location="cpu", weights_only=False)
        self.regressor.eval()

    def forward(self, img, img_ds, return_embedding: bool = False, scale_score: bool = True):
        f, _ = self.encoder(img)
        f_ds, _ = self.encoder(img_ds)
        f_combined = torch.hstack((f, f_ds))
        score = self.regressor(f_combined)
        if scale_score:
            score = self._scale_score(score)
        if return_embedding:
            return score, f_combined
        else:
            return score

    def _scale_score(self, score: float, new_range: Tuple[float, float] = (0., 1.)) -> float:
        # Compute scaling factors
        original_range = (available_datasets_ranges[self.regressor_dataset][0], available_datasets_ranges[self.regressor_dataset][1])
        original_width = original_range[1] - original_range[0]
        new_width = new_range[1] - new_range[0]
        scaling_factor = new_width / original_width

        # Scale score
        scaled_score = new_range[0] + (score - original_range[0]) * scaling_factor

        # Invert the scale if needed
        if available_datasets_mos_types[self.regressor_dataset] == "dmos":
            scaled_score = new_range[1] - scaled_score

        return scaled_score
