from statistics import mean
import cv2
import numpy as np
import os

def is_grayscale(img, threshold=20):
    im = cv2.imread(img) if isinstance(img, str) else img
    if len(im.shape) == 2:
        return True
    else:
        im = im.astype(np.float32)
        negative = (np.abs(im[:, :, 0] - im[:, :, 1]) > threshold).sum() + np.abs((im[:, :, 1] - im[:, :, 2]) > threshold).sum() 
        # if negative/np.prod(im.shape) < 0.3:
        nega_percent = negative/np.prod(im.shape)
        flag =  nega_percent < 0.01
        # if  flag:
        print(f"{img} {flag} {negative/np.prod(im.shape):.3f}")

        return  img, flag, nega_percent

# img_dir = 'img/grayscale'     # 0.001
# img_dir = 'img/brighterror'   # 0.08
# img_dir = 'img/angleerror'    # 0.138
# img_dir = 'img/occlude'         # 0.124
# img_dir = 'img/blur'         # 
img_dir = 'img/biterror'         # 0.036

img_list = os.listdir(img_dir)
# print(img_list)

percent_recorder = []
file_recoder = []
j = 0
for i in img_list:
    i = os.path.join(img_dir, i)
    img, flag, nega_percent = is_grayscale(i, threshold=15)
    percent_recorder.append(nega_percent)
    if flag: file_recoder.append(img)
    j += 1
    # if j > 40: break


print(f"分数范围： {min(percent_recorder):.3f}-{max(percent_recorder):.3f}, 均值为{mean(percent_recorder):.3f}")
print(f"筛除图片数量： {len(file_recoder)}/{j}")
for f in file_recoder:
    print(f)