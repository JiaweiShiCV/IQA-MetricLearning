import torch
from torch import nn
import torch.nn.functional as F

a = torch.randint(low=0, high=10, size=(2, 4))
b = torch.randint(low=0, high=10, size=(4, 4))

distmat = a*b
d = torch.mul(a, b)
print(y.shape)

# m = a*b.to(torch.float32)
# u = F.normalize(m, p=2, dim=1)
# print()