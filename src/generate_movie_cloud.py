from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import sklearn.datasets
import time
from PIL import ImageFile
from PIL import Image
from torch.utils.data import DataLoader
from sys import exit
# from Dynamic_Model import MDGAN_S1_G, MDGAN_S2_G, SubsetRandomSampler
from models.Dynamic_Model_attn_channel_cat import MDGAN_S1_G as OWN_S1_G
from models.Dynamic_Model_attn import MDGAN_S2_G as OWN_S2_G
from models.Dynamic_Model import MDGAN_S1_G, MDGAN_S2_G
from torchvision.utils import save_image
from data_loader.video_folder import get_val_DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True
cudnn.benchmark = True


parser = argparse.ArgumentParser()
name = ""

parser.add_argument('--dataset', type=str,
	default='cloud'
	)
parser.add_argument('--nframes', type=int,
	default=32
	)
parser.add_argument('--image_size', type=int,
	default=128
	)
parser.add_argument('--batch_size', type=int,
	default=16
	)
parser.add_argument('--num_workers', type=int,
	default=0
	)


own_stage_1_epoch = 60
parser.add_argument('--own_stage_1', 
	default='/host/space/horita-d/programing/python/conf/iccv2019/optical_flow/beach_udon_net/results_bmvc/4_21_cloud_my-model_stage1_hatc-8_v2_ssimlambda10/models/{}-BaseG.ckpt'.format(str(own_stage_1_epoch))
	)
own_stage_2_epoch = 12
parser.add_argument('--own_stage_2',
	default='/host/space/horita-d/programing/python/conf/iccv2019/optical_flow/beach_udon_net/results_bmvc/4_25_cloud_my-model_stage2v1.2/models/{}-RefineG.ckpt'.format(own_stage_2_epoch) 
	)
md_stage_1_epoch = 50
parser.add_argument('--md_stage_1',
	default='/host/space/horita-d/programing/python/conf/iccv2019/optical_flow/beach_udon_net/results_bmvc/4_23_cloud_MDGAN_stage1v1.1/models/{}-BaseG.ckpt'.format(md_stage_1_epoch)
	)
md_stage_2_epoch = 12
parser.add_argument('--md_stage_2',
	default='/host/space/horita-d/programing/python/conf/iccv2019/optical_flow/beach_udon_net/results_bmvc/4_27_cloud_MDGAN_stage2_hatc-1v1.4/models/{}-RefineG.ckpt'.format(md_stage_2_epoch)
	)
parser.add_argument('--data_test',
	default='/export/ssd/horita-d/dataset/sky_timelapse/sky_test'
	)

	
name += "{}".format(str(own_stage_1_epoch))
name += "_{}".format(str(own_stage_2_epoch))
name += "_{}".format(str(md_stage_1_epoch))
name += "_{}".format(str(md_stage_2_epoch))

parser.add_argument('--out_path', default = './results/cloud/{}'.format(name))
opt = parser.parse_args()

if not os.path.exists(opt.out_path):
	os.makedirs(opt.out_path)

localtime = time.asctime( time.localtime(time.time()) )
print('\n start new program! ')
print(localtime)

def restore_model(path, model, flag=False):
	if flag:
		state_dict = torch.load(path, map_location=lambda storage, loc: storage)
		from collections import OrderedDict
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
			name = k[7:] # remove `module.`
			new_state_dict[name] = v
		new_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
		model.load_state_dict(new_state_dict)
		return model
	else:
		new_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
		model.load_state_dict(new_state_dict)
		return model

def denorm(x):
	"""Convert the range from [-1, 1] to [0, 1]."""
	out = (x + 1) / 2
	return out.clamp_(0, 1)

imageSize = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

own_S1_G = nn.DataParallel(OWN_S1_G(ngf=32, sn=True)).to(device)
own_S1_G.eval()
own_S1_G = restore_model(opt.own_stage_1, own_S1_G)
own_S2_G = nn.DataParallel(OWN_S2_G(sn=True)).to(device).eval()
own_S2_G.eval()
own_S2_G = restore_model(opt.own_stage_2, own_S2_G)

md_S1_G = nn.DataParallel(MDGAN_S1_G(32)).to(device)
md_S1_G.eval()
md_S2_G = nn.DataParallel(MDGAN_S2_G(32)).to(device).eval()
md_S2_G.eval()

# Load models
md_S1_G = restore_model(opt.md_stage_1, md_S1_G, flag=True)
md_S2_G = restore_model(opt.md_stage_2, md_S2_G, flag=True)

# -------------set inputs---------------------------------

valid_loader = get_val_DataLoader(opt, None)

for i, (real_frames, _) in enumerate(valid_loader):
	out_path = "{}/{}".format(opt.out_path, i)
	if not os.path.exists(out_path):
		os.makedirs(out_path)

	real_frames = real_frames.to(device) # -> [B, 3, 32, 128, 128]
	first_frames = real_frames[:,:,0:1,:,:] # -> [2, 3, 128, 128]
	input_frames = first_frames.repeat(1,1,opt.nframes,1,1)

	print(input_frames.size())

	val_fake_own_s1 = OWN_S1_G(input_frames)
	val_fake_own_s2 = OWN_S2_G(input_frames)
	val_fake_md_s1 = MDGAN_S1_G(input_frames)
	val_fake_md_s2 = MDGAN_S2_G(input_frames)

	frames = torch.cat([real_frames.to(device), val_fake_own_s1, val_fake_own_s2, val_fake_md_s1, val_fake_md_s2], dim=4)
	

	for j in range(opt.nframes):
		a = frames[0][0][j].unsqueeze(0)
		b = frames[0][1][j].unsqueeze(0)
		c = frames[0][2][j].unsqueeze(0)
		A = torch.cat([a, b, c], dim=0)
		save_image(denorm(A.data.cpu()), '{}/{}.png'.format(out_path, i))


# save fake samples of stage 1
for t in range(val_fake_s1.size(0)):
	vutils.save_image(val_fake_s1[t],
						'%s/samples_s1_frame_%03d.png' 
						% (opt.outf, t),normalize=True, nrow = 8)                
# save fake samples of stage 2
for t in range(val_fake_s2.size(0)):
	vutils.save_image(val_fake_s2[t],
						'%s/samples_s2_frame_%03d.png' 
						% (opt.outf, t),normalize=True, nrow = 8)
# save real samples
for t in range(val_gt.permute(2,0,1,3,4).size(0)):
	vutils.save_image(val_gt.permute(2,0,1,3,4)[t],
						'%s/samples_real_frame_%03d.png'
						% (opt.outf, t),normalize=True, nrow = 8)    

def generate_video(model='s1', outf= opt.outf):
	img_path = os.path.join(outf, 'samples_' + model +  '_frame_%03d.png')
	mp4_path = os.path.join(outf, model+ '_video.mp4')
	cmd = ('ffmpeg -loglevel warning -framerate 25 -i ' + img_path + 
		' -qscale:v 2 -y ' + mp4_path )
	print(cmd)
	os.system(cmd)
generate_video('s1')
generate_video('s2')
generate_video('real')





# ffmpeg -loglevel warning -framerate 25 -i sample_%02.png -qscale:v 2 -y out.mp4