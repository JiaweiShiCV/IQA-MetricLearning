import torch.utils.data as data
import os
import cv2

class ImagePathDataset(data.Dataset):
    def __init__(self, dir, transforms=None, labelname_list=None):
        self.transforms = transforms
        self._labelname_list = labelname_list if labelname_list is not None else self.get_labelname_list(dir)
        self._class_num = len(self._labelname_list)
        self.label_dict = self.get_label_dict(labelname_list)
        self.files = self.get_fpaths(dir)
        self.files = self.filter_fpaths(self.files, self.label_dict)

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
    
    @property
    def class_num(self):
        return self._class_num

    @property
    def labelname_list(self):
        return self._labelname_list

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
    def filter_fpaths(paths, label_dict):
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
    def get_fpaths(dir):
        fpathlist = []
        for root, dirs, files in os.walk(dir, topdown=False):
            counter = 0
            for name in files:
                fpath = os.path.join(root, name)
                fpathlist.append(fpath)
                counter += 1
        return fpathlist
    
    @staticmethod
    def get_labelname_list(dir):
        return list(os.listdir(dir))

    @staticmethod
    def get_label_dict(labelname_list):
        label_dict = {}
        for idx, label in enumerate(labelname_list):
            label_dict[label] = idx
        return label_dict    
