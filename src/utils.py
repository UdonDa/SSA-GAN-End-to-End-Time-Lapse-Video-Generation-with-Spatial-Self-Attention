import os
import torch
import torch.nn as nn

# Create data loaders.
def load_path_array(dataset):
    if dataset == 'beach':
        txt = '../data/beach.txt'
    FILE = open(txt)
    FILES = FILE.readlines()
    FILE.close()
    FILES = [f.replace("\n", "") for f in FILES]
    return FILES

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

        
def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))


def generate_video(inputs=None, output_dir=None,  filename=None):
    img_path = os.path.join(inputs, '%05d.png')
    mp4_path = os.path.join(output_dir, f'{filename}.mp4')
    cmd = f"ffmpeg -y -v quiet -r 25 -i {img_path} -vcodec libx264 -pix_fmt yuv420p -r 60 {mp4_path} >> log.txt"
    os.system(cmd)