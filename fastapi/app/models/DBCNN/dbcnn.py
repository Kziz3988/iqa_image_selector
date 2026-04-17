import torch
import torchvision
import torch.nn as nn
from app.models.DBCNN.scnn import SCNN
import torch.nn.functional as F
from app.utils.paths import get_weight_path

class DBCNN(torch.nn.Module):

    def __init__(self, scnn_root):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Convolution and pooling layers of VGG-16.
        vgg16 = torchvision.models.vgg16(weights=None)
        vgg16.load_state_dict(torch.load(get_weight_path('vgg16.pth'), map_location=self.device))
        self.features1 = vgg16.features
        self.features1 = nn.Sequential(*list(self.features1.children())
                                            [:-1])
        scnn = SCNN().to(self.device)
              
        scnn.load_state_dict(torch.load(scnn_root, map_location=self.device))
        self.features2 = scnn.features
        
        # Linear classifier.
        self.fc = torch.nn.Linear(512*128, 1)
        
        # Freeze all previous layers.
        for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = False
        # Initialize the fc layers.
        nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias.data, val=0)

        

    def forward(self, X):
        """Forward pass of the network.
        """
        N = X.size()[0]
        X1 = self.features1(X)
        H = X1.size()[2]
        W = X1.size()[3]
        assert X1.size()[1] == 512
        X2 = self.features2(X)
        H2 = X2.size()[2]
        W2 = X2.size()[3]
        assert X2.size()[1] == 128        
        
        if (H != H2) | (W != W2):
            X2 = F.interpolate(X2, size=(H, W), mode='bilinear', align_corners=False)

        X1 = X1.view(N, 512, H*W)
        X2 = X2.view(N, 128, H*W)  
        X = torch.bmm(X1, torch.transpose(X2, 1, 2)) / (H*W)  # Bilinear
        assert X.size() == (N, 512, 128)
        X = X.view(N, 512*128)
        X = torch.sqrt(X + 1e-8)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 1)
        return X
