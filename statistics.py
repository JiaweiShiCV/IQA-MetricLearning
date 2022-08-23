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
        # print(batch.shape)
        features = model(batch, return_features=True)
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
    sub_dir = 'img/biterror'  

    dir = 'img'
    sub_dir_list = []
    for i in os.listdir(dir):
        sub_dir_list.append(os.path.join(dir, i))
    print(sub_dir_list)
    print()

    for sub_dir in sub_dir_list:
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1080, 1920)),
            transforms.ToTensor(),
        ])
        dataset = ImagePathDataset(sub_dir, data_transforms)
        data_loader = data.DataLoader(dataset, batch_size=8, num_workers=0)

        # extractor
        extractor = torch.jit.load('inception-2015-12-05.pt').eval()
        print(f"[{sub_dir}]: start extracting...")

        # extract feature
        stats = extract_features(extractor, data_loader).numpy()
        print("Extracting done.")

        # calculate mean & cov
        mean = np.mean(stats, 0)
        std = np.std(stats, 0)
        # cov = np.cov(stats, rowvar=False)
        info = {"mean": mean, 'std': std, }
        name = os.path.basename(sub_dir)

        with open(f"inception_{name}.pkl", "wb") as f:
            pickle.dump(info, f)
        print(info)
        print("Info saved.")
        print()

