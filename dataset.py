import imp
import torch.utils.data as data
import os
import cv2

class ImagePathDataset(data.Dataset):
    def __init__(self, dir, label_dict, transforms=None):
        self.transforms = transforms
        self.label_dict = label_dict
        self.files = self.get_fpathlist(dir)
        self.files = self.check_paths(self.files, self.label_dict)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        path = self.files[i]
        img = cv2.imread(path)[:, :, ::-1]
        if self.transforms is not None:
            img = self.transforms(img)
        labelname = self.get_labelname(path)
        label = self.get_label(labelname, self.label_dict)
        return {
            'img': img,
            'label': label,
            'path': path,
        } 
    
    @staticmethod
    def check(path, label_dict):
        labelname = ImagePathDataset.get_labelname(path)
        return path if labelname in label_dict else None
    
    @staticmethod
    def get_labelname(path):
        return os.path.dirname(path).split('\\')[-1]  # windows下路径'\'

    @staticmethod
    def get_label(labelname, label_dict):
        return label_dict[labelname]
    
    @staticmethod
    def check_paths(paths, label_dict):
        # from multiprocessing import Pool
        # def check(path):
        for idx, path in enumerate(paths):
            labelname = ImagePathDataset.get_labelname(path)
            if labelname not in label_dict: 
                paths[idx] = None

        # pool = Pool(4)
        # paths = pool.map(ImagePathDataset.check, paths, list(label_dict*len(paths)))
        paths = list(filter(None, paths))
        # pool.close()
        # pool.join()
        return paths

    @staticmethod
    def get_fpathlist(dir):
        fpathlist = []
        for root, dirs, files in os.walk(dir, topdown=False):
            counter = 0
            for name in files:
                fpath = os.path.join(root, name)
                fpathlist.append(fpath)
                counter += 1
        return fpathlist
    
