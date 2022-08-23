import torch
import cv2
import os
import pickle
from PIL import Image
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix
from multiprocessing import Pool
from dataset import ImagePathDataset


# 设置类别标签
labelname_list = ['grayscale', 'brighterror', 'angleerror', 'occlude', 'blur', 'biterror']
label_dict = {}
for idx, label in enumerate(labelname_list):
    label_dict[label] = idx
print(label_dict)

mean, std = [], []
for name in labelname_list:
    with open(f"inception_{name}.pkl", 'rb') as f:
        info = pickle.load(f)
    mean.append(info['mean'][None, :]) 
    std.append(info['std'][None, :])
mean = torch.from_numpy(np.concatenate(mean, axis=0))
std = torch.from_numpy(np.concatenate(std, axis=0))
# print(mean.shape, std.shape)


if __name__ == "__main__":
    dir = 'img'
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1080, 1920)),
        transforms.ToTensor(),
    ])

    dataset = ImagePathDataset(dir, label_dict, data_transforms)
    print(len(dataset))
    # dataset.check_paths()
    print(f"Total num: {len(dataset)}.")
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    # model
    extractor = torch.jit.load('inception-2015-12-05.pt').eval()

    labels, preds, paths = [], [], []
    counter = 0
    for idx, batch in enumerate(dataloader):
        
        img_batch = batch['img']
        label_batch = batch['label']
        path_batch = batch['path']

        counter += img_batch.size(0)

        features = extractor(img_batch, return_features=True)

        features_repeat = torch.repeat_interleave(features[:, None, :], repeats=6, dim=1)
        distmat = features_repeat - mean
        distmat = torch.norm(distmat, dim=2, p=2) #/ (std+1e-9)
        _, pred_batch = torch.min(distmat, dim=1)
        labels.append(label_batch)
        preds.append(pred_batch)
        paths.extend(path_batch)
        print(f"{idx}, {counter}")

    labels = torch.cat(labels)
    preds = torch.cat(preds)

    acc = (preds == labels).sum() / counter
    print(acc)

    cm = np.array(confusion_matrix(np.array(labels), np.array(preds)))

    # labels_name = os.listdir(dir)
    plot_confusion_matrix(cm, labelname_list, "CMatrix")


