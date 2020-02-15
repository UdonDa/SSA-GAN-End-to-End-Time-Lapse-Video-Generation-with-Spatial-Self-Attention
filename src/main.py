from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time
from PIL import ImageFile
from PIL import Image
from data_loader.video_folder import get_train_DataLoader, get_val_DataLoader
from torch.utils.data import DataLoader
from config.parameter import get_parameters
from solver_beach import Solver as SolverBeach
import sys
import random
from utils import load_path_array

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

def main(config):
    os.makedirs(config.main_path, exist_ok=True)
    os.makedirs(config.gif_path, exist_ok=True)
    os.makedirs(config.image_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)
    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.sample_path, exist_ok=True)
    os.makedirs(config.val_path, exist_ok=True)
    os.makedirs(config.metric_path, exist_ok=True)

    with open('{}/config.txt'.format(config.main_path), 'w') as f:
        print(config, file=f)

    train_path, eval_path = None, None
    # if config.dataset == 'beach':
    #     path_array = load_path_array(config.dataset)
    #     # random.shuffle(path_array)
    #     boundary1 = int(len(path_array) * 0.1)
    #     boundary2 = int(len(path_array) * 0.2)
    #     train_path, eval_path = path_array[0:boundary1], path_array[boundary1:boundary2]

    #     # For evaluation
    #     random.shuffle(eval_path)
    #     eval_path = eval_path[:1000]

    if config.mode == 'train':
        train_loader = get_train_DataLoader(config, train_path)
        val_loader = get_val_DataLoader(config, eval_path)
        solver = SolverBeach(train_loader, val_loader, config)
        if not config.stage2:
            print("Start train stage 1.")
            solver.train_stage1()
        elif config.stage2:
            print("Start train stage 2.")
            solver.train_stage2()

    elif config.mode == "save_val_img":
        val_loader = get_val_DataLoader(config, eval_path)
        solver = SolverBeach(None, val_loader, config)
        solver.generate_val_samples(val_loader)

    elif config.mode == 'save_img':
        
        print("Start save images mode.")

        transform = transforms.Compose([
                            transforms.Resize((128 , 128)),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                (0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5)),
                        ])
        dataset = ImageFolder(
            "./test_inputs/",
            transform=transform
        )
        loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)
        solver = SolverBeach(loader, loader, config)
        solver.test(loader)


if __name__ == '__main__':
    config = get_parameters()
    torch.backends.cudnn.benchmark=True
    main(config)
