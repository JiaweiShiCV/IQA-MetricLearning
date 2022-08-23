from skimage import measure, io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import os
import cv2


def cal_entroy(img_path):
    img = io.imread(img_path)
    rawimg = img

    img = rgb2gray(img)     # 转灰度图
    print(img)
    print(img.shape)
    fig, axes = plt.subplots(1,2, figsize=(8, 4))
    ax = axes.ravel()
    ax[0].imshow(rawimg)
    ax[1].imshow(img, cmap=plt.cm.gray)
    plt.show()
    entroy = measure.shannon_entropy(img)
    img_name = os.path.basename(img_path).split('.')[0]
    print(f'{img_name:15}: {entroy:.2f}')
    return entroy

creatVar = locals()
img_list = [
    # './img/grayscale.jpg',
    './img/sky.jpg',
    './img/overexposure.jpg',
    # './img/night.jpg',
    # './img/normal.jpg',
    # './img/blur.jpg'
]

for index, i in enumerate(img_list):
    # creatVar[f'img{index}'] = i
    cal_entroy(i)


