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
labelname_list = ['angleerror', 'biterror', 'blur', 'brighterror', 'close', 'far', 'grayscale', 'occlude',]

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

    dataset = ImagePathDataset(dir, data_transforms, labelname_list)
    print(len(dataset))
    # dataset.check_paths()
    print(f"Total num: {len(dataset)}.")
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    # model
    extractor = torch.jit.load('inception-2015-12-05.pt').eval()

    features, labels, preds, paths = [], [], [], []
    counter = 0
    for idx, batch in enumerate(dataloader):
        
        img_batch = batch['img']
        label_batch = batch['label']
        path_batch = batch['path']
        counter += img_batch.size(0)

        feature_batch = extractor(img_batch, return_features=True)

        feature_batch_repeat = torch.repeat_interleave(feature_batch[:, None, :], repeats=dataset.class_num(), dim=1)
        distmat = (feature_batch_repeat - mean) / (std+1e-9) 
        distmat = torch.norm(distmat, dim=2, p=2) 
        _, pred_batch = torch.min(distmat, dim=1)

        features.append(feature_batch)
        labels.append(label_batch)
        preds.append(pred_batch)
        paths.extend(path_batch)
        print(f"{idx}, {counter}")

    features = torch.cat(features)
    labels = torch.cat(labels)
    preds = torch.cat(preds)
    torch.save(
        {'feat': features, 'label': labels, 'pred': preds, 'path': paths},
        'points.pt'
    )

    acc = (preds == labels).sum() / counter
    print(acc)

    cm = np.array(confusion_matrix(np.array(labels), np.array(preds)))

    # labels_name = os.listdir(dir)
    plot_confusion_matrix(cm, labelname_list, f"CMatrix-acc{acc:.2f}", acc)


