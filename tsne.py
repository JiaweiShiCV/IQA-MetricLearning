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
    # dir = 'img'
    # data_transforms = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((1080, 1920)),
    #     transforms.ToTensor(),
    # ])

    # dataset = ImagePathDataset(dir, label_dict, data_transforms)
    # print(len(dataset))
    # # dataset.check_paths()
    # print(f"Total num: {len(dataset)}.")
    # dataloader = data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    # # model
    # extractor = torch.jit.load('inception-2015-12-05.pt').eval()

    # labels, preds, paths, features = [], [], [], []
    # counter = 0
    # for idx, batch in enumerate(dataloader):
        
    #     img_batch = batch['img']
    #     label_batch = batch['label']
    #     path_batch = batch['path']

    #     counter += img_batch.size(0)

    #     feature_batch = extractor(img_batch, return_features=True)

    #     # features_repeat = torch.repeat_interleave(feature_batch[:, None, :], repeats=6, dim=1)
    #     # distmat = (features_repeat - mean) / (std+1e-9) 
    #     # distmat = torch.norm(distmat, dim=2, p=2) 
    #     # _, pred_batch = torch.min(distmat, dim=1)
    #     features.append(feature_batch)
    #     labels.append(label_batch)
    #     # preds.append(pred_batch)
    #     # paths.extend(path_batch)
    #     print(f"{idx}, {counter}")

    # features = torch.cat(features)
    # labels = torch.cat(labels)
    # torch.save(
    #     {'feat': features, 'label': labels},
    #     'points.pt'
    # )
    # preds = torch.cat(preds)

    # acc = (preds == labels).sum() / counter
    # print(acc)

    # cm = np.array(confusion_matrix(np.array(labels), np.array(preds)))

    # # labels_name = os.listdir(dir)
    # plot_confusion_matrix(cm, labelname_list, "CMatrix")
    points = torch.load('points.pt', map_location='cpu')
    features = points['feat'].numpy()
    labels = points['label'].numpy()

    plot_TSNE(features, labels, labelname_list, 2)



# 19个类别，每个类别2048维特征长度
# features = torch.rand(19, 2048)

