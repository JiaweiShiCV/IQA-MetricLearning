from torchvision.io import read_image
from model import *
from dataset import ImagePathDataset
from torchvision.models import ResNet50_Weights


dir = 'normal'

files = ImagePathDataset.get_fpaths(dir)

extractor = ResNet50()
transform = ResNet50_Weights.DEFAULT.transforms()

for f in files:
    img = read_image(f)
    img = transform(img)[None]
    feat = extractor(img)
    print(feat)



