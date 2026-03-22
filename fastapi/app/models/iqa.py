import torch
from PIL import Image
from torchvision import transforms
from app.models.VCRNet.IQASolver import demoIQA as VCRNet
from app.models.MANIQA.maniqa import IQAModule as MANIQA, ImageData
from app.models.MANIQA.config import Config
from app.models.MANIQA.utils.inference_process import ToTensor, Normalize
from tqdm import tqdm
from app.models.ARNIQA.arniqa import ARNIQA
from app.models.ARNIQA.utils.utils_data import center_corners_crop
from app.utils.paths import get_weight_path

class BaseModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    def predict(self, img_path):
        raise NotImplementedError  

class VCRNetPipeline(BaseModel):
    def __init__(self):
        super().__init__()
        
        self.model = VCRNet().to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model(img_tensor)
        return float(score.item())
    
class MANIQAPipeline(BaseModel):
    def __init__(self):
        super().__init__()
        self.config = Config({
            "num_crops": 1,
            "patch_size": 8,
            "img_size": 224,
            "embed_dim": 768,
            "dim_mlp": 768,
            "num_heads": [4, 4],
            "window_size": 4,
            "depths": [2, 2],
            "num_outputs": 1,
            "num_tab": 2,
            "scale": 0.8,
            "ckpt_path": 'ckpt_koniq10k.pt',
        })
        self.model = MANIQA(self.config).to(self.device)
        model_weights_path = get_weight_path(self.config.ckpt_path)
        self.model.net.load_state_dict(torch.load(model_weights_path, map_location="cpu", weights_only=False))
        self.model.eval()
        self.transform = transforms.Compose([Normalize(0.5, 0.5), ToTensor()])

    def predict(self, img_path):
        image = ImageData(image_path=img_path, transform=self.transform, num_crops=self.config.num_crops)
        avg_score = 0
        for i in tqdm(range(self.config.num_crops)):
            with torch.no_grad():
                patch_sample = image.get_patch(i)
                patch = patch_sample['d_img_org'].to(self.device)
                patch = patch.unsqueeze(0)
                score = self.model(patch)
                avg_score += score
        avg_score /= self.config.num_crops
        return avg_score.item()
    
class ARNIQAPipeline(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = ARNIQA().to(self.device)
        self.model.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def predict(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)
        img = center_corners_crop(img, crop_size=224)
        img_ds = center_corners_crop(img_ds, crop_size=224)
        
        img = [transforms.ToTensor()(crop) for crop in img]
        img = torch.stack(img, dim=0)
        img = self.normalize(img).to(self.device)
        img_ds = [transforms.ToTensor()(crop) for crop in img_ds]
        img_ds = torch.stack(img_ds, dim=0)
        img_ds = self.normalize(img_ds).to(self.device)

        with torch.no_grad():
            score = self.model(img, img_ds)
        return float(score.mean(0).item())

class IQAFactory:
    @staticmethod
    def get_model(name='VCRNet'):
        if name == 'VCRNet':
            return VCRNetPipeline()
        elif name == 'MANIQA':
            return MANIQAPipeline()
        elif name == 'ARNIQA':
            return ARNIQAPipeline()
        else:
            raise ValueError(f"Unknown IQA model: {name}")