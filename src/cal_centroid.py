import torch
import cv2
import os
import pickle
from PIL import Image
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from dataset import ImagePathDataset


def extract_features(model, loader):
    stats = []
    for idx, batch in enumerate(loader):
        features = model(batch['img'], return_features=True)
        stats.append(features)
    stats = torch.cat(stats, dim=0)
    return stats


if __name__ == "__main__":
    device = "cpu"  #"cuda"

    # data
    # sub_dir = 'img/grayscale'     
    # sub_dir = 'img/brighterror'   
    # sub_dir = 'img/angleerror'    
    # sub_dir = 'img/occlude'         
    # sub_dir = 'img/blur'       
    # sub_dir = 'img/biterror'  

    dir = 'img'
    sub_dir_list = []
    for i in os.listdir(dir):
        sub_dir_list.append(os.path.join(dir, i))
    print(sub_dir_list)
    print()

    labelname_list = ImagePathDataset.get_labelname_list(dir)

    for sub_dir in sub_dir_list:
        print(f"[{sub_dir}]: start extracting...")
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1080, 1920)),
            transforms.ToTensor(),
        ])
        dataset = ImagePathDataset(sub_dir, data_transforms, labelname_list)
        print(f"{sub_dir:8} image number: {len(dataset)}")
        data_loader = data.DataLoader(dataset, batch_size=8, num_workers=0)

        # extractor
        extractor = torch.jit.load('inception-2015-12-05.pt').eval()

        # extract feature
        stats = extract_features(extractor, data_loader).numpy()
        print("Extracting done.")

        # calculate mean & cov
        mean = np.mean(stats, 0)
        std = np.std(stats, 0)
        # cov = np.cov(stats, rowvar=False)
        info = {"mean": mean.shape, 'std': std.shape, }
        name = os.path.basename(sub_dir)

        with open(f"inception_{name}.pkl", "wb") as f:
            pickle.dump(info, f)
        print(info)
        print("Info saved.")
        print()

