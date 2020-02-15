# python eval.py --metrics ssim --metrics psnr --metrics mse --exp_iter 200000


# Thanks to https://github.com/TUM-LMF/FutureGAN/blob/master/eval.py
'''
Script to
    1. generate and evaluate test samples of FutureGAN, or
    2. calculate evaluation metrics for existing test samples of a baseline model (`CopyLast`, `fRNN`, `MCNet`).
-------------------------------------------------------------------------------
1. To generate and evaluate test samples of FutureGAN, please set the --model_path flag correctly:
    --model_path=`path_to_FutureGAN_generator_ckpt`
Your test data to generate predictions and evaluate FutureGAN is assumed to be arranged in this way:
    data_root/video(n)/frame(m).ext
    n corresponds to number of video folders, m to number of frames in eachfolder.
For evaluation you can choose which metrics are calculated, please set the --metrics flag accordingly.
Your choices are: `mse`, `psnr`, `ssim`, `ssim2`, `ms_ssim`.
If you want to calculate multiple metrics, simply append them using the --metrics flag:
    --metrics=`metric1` --metrics=`metric2` ...
-------------------------------------------------------------------------------
2. To calculate evaluation metrics for existing test samples of a baseline model, please set the --model flag correctly:
    --baseline=`shortname_of_baseline_model`, one of: `CopyLast`, `fRNN`, `MCNet`
Your data to evaluate the results of `CopyLast` baseline is assumed to be arranged in this way:
    data_root/video(n)/frame(m).ext
    n corresponds to number of video folders, m to number of frames in eachfolder.
Your data to evaluate the results of a baseline model other than `CopyLast` is assumed to be arranged in this way:
    Ground truth frames:
        data_root/in_gt/video(n)/frame(m).ext
    Predicted frames
        data_root/in_pred/video(n)/frame(m).ext
    n corresponds to number of video folders, m to number of frames in eachfolder.
For evaluation you can choose which metrics are calculated, please set the --metrics flag accordingly.
Your choices are: `mse`, `psnr`, `ssim`, `ssim2`, `ms_ssim`.
If you want to calculate multiple metrics, simply append them using the --metrics flag:
    --metrics=`metric1` --metrics=`metric2` ...
-------------------------------------------------------------------------------
For further options and information, read the provided `help` information of the optional arguments below.
'''

import os
import time
import argparse
from PIL import Image
from math import floor, ceil
import numpy as np
import imageio
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
from eval_utils import get_image_grid, save_image_grid, count_model_params
from models.model_new_BaseNet_with_attn import BaseNetGenerator
import eval_metrics as eval_metrics
from data_loader.video_folder import get_val_DataLoader
from torch.utils.data import DataLoader, sampler
from sys import exit


def str2bool(v):
    return v.lower() in ('true')

# =============================================================================
# config options

help_description = 'This script evaluates a FutureGAN model or one of these baseline models: `CopyLast` `fRNN` `MCNet`, according to the specified arguments.'

parser = argparse.ArgumentParser(description=help_description)

# general
parser.add_argument('--random_seed', type=int, default=int(time.time()), help='seed for generating random numbers, default = `int(time.time())`')
parser.add_argument('--ext', action='append', default=['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm'], help='list of strings of allowed file extensions, default=[.jpg, .jpeg, .png, .ppm, .bmp, .pgm]')

parser.add_argument('--exp_name', type=str, default='exp_LR0202_LR0509_128_2', help='path to FutureGAN`s generator checkpoint, default=``')
parser.add_argument('--exp_iter', type=int, default=300000, help='model iter num')

parser.add_argument('--model_type', type=str, default='own', help='model to calculate evaluation metrics for (choices: `own`, `CopyLast`, `MCNet`, `fRNN`), default=`own`')
parser.add_argument('--data_test', type=str, default='/export/ssd/horita-d/dataset/sky_timelapse/sky_test', help='path to root directory of test data (ex. -->path_to_dataset/test)')
parser.add_argument('--test_dir', type=str, default='./tests', help='path to directory for saving test results, default=`./tests`')
parser.add_argument('--experiment_name', type=str, default='', help='name of experiment, default=``')

parser.add_argument('--nc', type=int, default=3, help='number of input image channels, default=3')
parser.add_argument('--image_size', type=int, default=128, help='frame resolution, default=128')
parser.add_argument('--nframes', type=int, default=32, help='number of video frames to generate or predict for one sample, default=6')
parser.add_argument('--deep_pred', type=int, default=1, help='number of (recursive) prediction steps for future generator in test mode, default=1')
parser.add_argument('--batch_size', type=int, default=16, help='batch size at test time, change according to available gpu memory, default=8')
parser.add_argument('--metrics', action='append', help='list of evaluation metrics to calculate (choices: `mse`, `psnr`, `ssim`, `ssim2`), default=``')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--optical_flow', type=str2bool, default=False)

# display and save
parser.add_argument('--save_frames_every', type=int, default=1, help='save video frames every specified iteration, default=1')
parser.add_argument('--save_gif_every', type=int, default=1, help='save gif every specified iteration, default=1')
parser.add_argument('--in_border', type=str, default='black', help='color of border added to gif`s input frames (`color_name`), default=`black`')
parser.add_argument('--out_border', type=str, default='red', help='color of border added to gif`s output frames (`color_name`), default=`red`')
parser.add_argument('--npx_border', type=int, default=2, help='number of border pixels, default=2')

# parse and save training config
config = parser.parse_args()

torch.set_default_tensor_type('torch.cuda.FloatTensor')


# ===================================================================================
def evaluate_pred(config):
    # Configs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_path = config.data_test
    nframes = config.nframes
    model_type = config.model_type
    exp_name = config.exp_name
    exp_iter = config.exp_iter
    metric_path = '../results_bmvc/{}/metric'.format(exp_name)
    data_loader = None

    # Create metric save path.
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)

    # Load Generator model.
    if model_type == 'own':
        G = BaseNetGenerator().to(device)
        G = nn.DataParallel(G)
        G_path = '../results_bmvc/{}/models/{}-BaseG.ckpt'.format(exp_name, exp_iter)
        G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        # G.eval()

    # Load test dataset.
    if model_type == 'own':
        data_loader = get_val_DataLoader(config)
        
    # Save test configuration.
    if True:
        with open('../results_bmvc/{}/metric/eval_config_{}.txt'.format(exp_name, exp_iter), 'w') as f:
            print('------------- test configuration -------------', file=f)
            for l, m in vars(config).items():
                print(('{}: {}').format(l, m), file=f)
            print(' ... loading test configuration ... ')
            print(' ... saving test configuration {}'.format(f))

    # Define metrics.
    if config.metrics is not None:
        metrics_values = {}
        for metric_name in config.metrics:
            metrics_values['{}_frames'.format(metric_name)] = torch.zeros_like(torch.FloatTensor(len(data_loader) * config.batch_size, nframes))
            metrics_values['{}_avg'.format(metric_name)] = torch.zeros_like(torch.FloatTensor(len(data_loader) * config.batch_size,1))
            print(' ... calculating {} ...'.format(metric_name))

    if config.metrics is not None:
        metrics_i_video = {}
        for metric_name in config.metrics:
            metrics_i_video['{}_i_video'.format(metric_name)] = 0

    if True:
        data_iter = iter(data_loader)
        for i in tqdm(range(len(data_iter))):
            real_frames, _ = next(data_iter)
            real_frames = real_frames.to(device)
            input_frames = real_frames[:,:,0:1,:,:]
            input_frames = input_frames.repeat(1, 1, config.nframes, 1, 1)

            # Generate fake frames.
            with torch.no_grad():
                fake_frames = G(input_frames.detach())
            
            # Evaluate fake frames.
            if config.metrics is not None:
                for metric_name in config.metrics:
                    calculate_metric = getattr(eval_metrics, 'calculate_{}'.format(metric_name))

                    for i_batch in range(fake_frames.size(0)):
                        for i_frame in range(nframes):
                            metrics_values['{}_frames'.format(metric_name)][metrics_i_video['{}_i_video'.format(metric_name)], i_frame] = calculate_metric(fake_frames[i_batch,:,i_frame,:,:], real_frames[i_batch,:,i_frame,:,:])
                            metrics_values['{}_avg'.format(metric_name)][metrics_i_video['{}_i_video'.format(metric_name)]] = torch.mean(metrics_values['{}_frames'.format(metric_name)][metrics_i_video['{}_i_video'.format(metric_name)]])
                        metrics_i_video['{}_i_video'.format(metric_name)] = metrics_i_video['{}_i_video'.format(metric_name)] + 1


    # calculate and save mean eval statistics
    if config.metrics is not None:
        metrics_mean_values = {}
        for metric_name in config.metrics:
            test_dir = './eval_result'

            metrics_mean_values['{}_frames'.format(metric_name)] = torch.mean(metrics_values['{}_frames'.format(metric_name)],0)
            metrics_mean_values['{}_avg'.format(metric_name)] = torch.mean(metrics_values['{}_avg'.format(metric_name)],0)            
            torch.save(metrics_mean_values['{}_frames'.format(metric_name)], os.path.join(test_dir, '{}_frames.pt'.format(metric_name)))
            torch.save(metrics_mean_values['{}_avg'.format(metric_name)], os.path.join(test_dir, '{}_avg.pt'.format(metric_name)))

        print(' ... saving evaluation statistics to dir: {}'.format(test_dir))
        print('============= result ===========')
        print(metrics_mean_values)
        with open('../results_bmvc/{}/metric/eval_config_{}.txt'.format(exp_name, exp_iter), 'w') as f:
            print('------------- test configuration -------------', file=f)
            for l, m in vars(config).items():
                print(('{}: {}').format(l, m), file=f)
            print(' ... loading test configuration ... ')
            print(' ... saving test configuration {}'.format(f))
        




if __name__ == '__main__':
    evaluate_pred(config)



