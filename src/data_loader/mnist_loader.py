import torch
from torchvision import datasets
import numpy as np

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample
    

dataset = datasets.DatasetFolder(
    root='/export/ssd/horita-d/dataset/moving_gan/mnist_test_seq.npy',
    loader=npy_loader,
    extensions=['.npy']
)
