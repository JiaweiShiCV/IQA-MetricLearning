from turtle import forward
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch



class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=self.weights)
        self.extracter = nn.Sequential(*list(resnet.children())[:-1])
    
    def transform(self):
        return self.weights.transforms()

    def forward(self, x, return_features=True):
        x = self.extracter(x)

        return x[:, :, 0, 0]


if __name__ == "__main__":
    model = ResNet50()
    x = torch.randn(1, 3, 224, 224)
    x = model(x)
    print(x.shape)
    pass


