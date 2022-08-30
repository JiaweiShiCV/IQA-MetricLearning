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


def plot_TSNE(feature, label, pred, labelname_list, paths, dims=2, enable_pandasgui=True):
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
        data = np.concatenate(
            (x[:, None], y[:, None], label[:, None]+1, pred[:, None]+1, labelnames[:, None], paths[:, None]),
            axis=1,
        )
        df = pd.DataFrame(
            data, 
            columns=['x', 'y', 'labelnum', 'pred', 'labelname', 'paths'],
        )
        df = df.astype({'x': 'float', 'y': 'float', 'pred': 'float', 'labelnum': 'float'})
        show(df)


if __name__ == "__main__":
    points = torch.load('points/points_neighbor_euclidean_ResNet50.pt', map_location='cpu')
    features = points['feat'].numpy()
    labels = points['label'].numpy()
    preds = points['pred'].numpy()
    paths = np.array(points['path']) 
    labelname_list = points['labelname_list']

    label_dict = ImagePathDataset.get_label_dict(labelname_list)
    print(label_dict)
    
    plot_TSNE(features, labels, preds, labelname_list, paths, 2)


