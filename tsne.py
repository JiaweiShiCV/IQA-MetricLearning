import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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
import pandas as pd
from pandasgui import show


def plot_TSNE(feature, label, labelname_list, dims=2, enable_pandasgui=True):
    tsne = TSNE(n_components=dims)
    tsne.fit_transform(feature)

    # plot
    x = tsne.embedding_[:, 0]
    y = tsne.embedding_[:, 1]
    labelnames = np.array(labelname_list)[label]

    plt.figure()
    if tsne.embedding_.shape[1] == 2:
        ax = plt.gca()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        scatter = ax.scatter(x, y, c=label, s=(label+10)*2, alpha=0.5)
        # for i in range(len(x)):
        #     ax.text(x[i],y[i], label[i])
    elif tsne.embedding_.shape[1] == 3:
        z = tsne.embedding_[:, 2]
        ax = plt.gca(projection='3d')
        ax.set_ylabel('z')
        ax.scatter(x, y, z, c=label, s=20, alpha=0.5)
        # for i in range(len(x)):
        #     ax.text(x[i],y[i],z[i],i)
    
    plt.legend(handles=scatter.legend_elements()[0], labels=labelname_list)
    plt.savefig('./plots/tsne.png')
    
    if enable_pandasgui:
        data = np.concatenate((x[:, None], y[:, None], label[:, None]+2, labelnames[:, None]), axis=1)
        df = pd.DataFrame(data, columns=['x', 'y', 'labelnum', 'labelname'])
        df = df.astype({'x': 'float', 'y': 'float', 'labelnum': 'float'})
        show(df)

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
    points = torch.load('points.pt', map_location='cpu')
    features = points['feat'].numpy()
    labels = points['label'].numpy()

    plot_TSNE(features, labels, labelname_list, 2)


