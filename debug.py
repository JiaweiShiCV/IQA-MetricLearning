import pickle
import torch
import numpy as np

mean, std = [], []
for name in ['grayscale', 'brighterror', 'angleerror', 'occlude', 'blur', 'biterror']:
    with open(f"inception_{name}.pkl", 'rb') as f:
        info = pickle.load(f)
    mean.append(info['mean'][None, :]) 
    std.append(info['std'][None, :])
mean = torch.from_numpy(np.concatenate(mean, axis=0))
std = torch.from_numpy(np.concatenate(std, axis=0))
print(mean.shape, std.shape)
