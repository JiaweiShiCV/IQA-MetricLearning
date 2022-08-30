import torch
import cv2
import os
import pickle
from PIL import Image
import tqdm
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix, nearest_centroid, nearest_neighbor
from multiprocessing import Pool
from dataset import ImagePathDataset
from model import *
import time


def inference(dataloader, extractor, ):
    extractor.eval()
    features, labels, paths = [], [], []
    counter = 0
    with torch.no_grad():
        for idx, batch in (enumerate(dataloader)):
            cur = time.time()
            img_batch = batch['img']
            label_batch = batch['label']
            path_batch = batch['path']
            counter += img_batch.size(0)

            feature_batch = extractor(img_batch, return_features=True)
            features.append(feature_batch)
            labels.append(label_batch)
            paths.extend(path_batch)
            cost=time.time()-cur
            print(f"{idx}, {counter}, {cost}")

    features = torch.cat(features)
    labels = torch.cat(labels)

    info = {
        'feat': features,
        'label': labels,
        'path': paths,
        'labelname_list': dataloader.dataset.labelname_list
    }

    return info

def prediction(info, algorithm=None):
    features = info['feat']
    labels = info['label']
    labelname_list = info['labelname_list']

    if algorithm == 'nearest_centroid':
        preds = nearest_centroid(features, labelname_list)
    elif algorithm == 'nearest_neighbor':
        preds = nearest_neighbor(features, features, labels)

    acc = (preds == labels).sum() / preds.shape[0]
    info.update({'acc': acc, 'pred': preds})

    return info


if __name__ == "__main__":
    # Building model
    # extractor = torch.jit.load('inception-2015-12-05.pt')
    extractor = ResNet50()

    # Label names
    labelname_list = ['angleerror', 'biterror', 'blur', 'brighterror', 'close', 'far', 'grayscale', 'occlude', 'normal']

    # Directory
    dir = 'img'

    # Preparing data
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1080, 1920)),
        transforms.ToTensor(),
    ])
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        extractor.transform()
    ])

    dataset = ImagePathDataset(dir, data_transforms, labelname_list)
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    print(f"Total num: {len(dataset)}.")

    # Inference and prediction
    info = inference(dataloader, extractor, )
    info = prediction(info, algorithm='nearest_neighbor')

    # Saving information
    torch.save(
        info,
        f'points/points_neighbor_euclidean_{extractor.__class__.__name__}.pt'
    )

    # Ploting confusion matrix
    labels = info['label']
    preds = info['pred']
    acc = info['acc']
    print(acc)
    cm = np.array(confusion_matrix(np.array(labels), np.array(preds)))
    plot_confusion_matrix(cm, labelname_list, f"CMatrix-acc{acc:.4f}", acc)


