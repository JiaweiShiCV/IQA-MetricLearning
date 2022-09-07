from skimage import measure, io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import os
import cv2
from dataset import ImagePathDataset


def cal_entroy(img_path):
    img = io.imread(img_path)
    rawimg = img

    # img = rgb2gray(img)     # 转灰度图
    # print(img)
    # print(img.shape)
    # fig, axes = plt.subplots(1,2, figsize=(8, 4))
    # ax = axes.ravel()
    # ax[0].imshow(rawimg)
    # ax[1].imshow(img, cmap=plt.cm.gray)
    # plt.show()
    entroy = measure.shannon_entropy(img)
    img_name = os.path.basename(img_path)
    if entroy < 7:
        print(f'{img_path:25}: {entroy:.2f}')
        save_path = os.path.join('picked', img_name)
        io.imsave(save_path, img)
    return entroy

dir = 'img'
img_paths = ImagePathDataset.get_fpaths(dir)
print(img_paths.__len__())


for index, i in enumerate(img_paths):
    # creatVar[f'img{index}'] = i
    cal_entroy(i)


