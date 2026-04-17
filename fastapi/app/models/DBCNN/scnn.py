import torch.nn as nn
from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def weight_init(net): 
    for m in net.modules():    
        if isinstance(m, nn.Conv2d):         
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        


class SCNN(nn.Module):

    def __init__(self):
        """Declare all needed layers."""
        super(SCNN, self).__init__()

        # Linear classifier.

        self.num_class = 39
#        self.features = nn.Sequential(nn.Conv2d(3,48,3,1,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(48,48,3,2,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(48,64,3,1,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(64,64,3,2,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(64,64,3,1,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(64,64,3,2,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(64,128,3,1,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(128,128,3,1,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(128,128,3,2,1),nn.ReLU(inplace=True))
        self.features = nn.Sequential(nn.Conv2d(3,48,3,1,1),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
                                      nn.Conv2d(48,48,3,2,1),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
                                      nn.Conv2d(48,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,3,2,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        weight_init(self.features)
        self.pooling = nn.AvgPool2d(14,1)
        self.projection = nn.Sequential(nn.Conv2d(128,256,1,1,0), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                        nn.Conv2d(256,256,1,1,0), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        weight_init(self.projection)    
        self.classifier = nn.Linear(256,self.num_class)
        weight_init(self.classifier)

    def forward(self, X):
#        return X
        N = X.size()[0]
        assert X.size() == (N, 3, 224, 224)
        X = self.features(X)
        assert X.size() == (N, 128, 14, 14)
        X = self.pooling(X)
        assert X.size() == (N, 128, 1, 1)
        X = self.projection(X)
        X = X.view(X.size(0), -1)          
        X = self.classifier(X)
        assert X.size() == (N, self.num_class)
        return X
