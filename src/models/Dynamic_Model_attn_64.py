# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

# Dynamic_Model.py
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
try:
    from . import self_attention_for_hw
except:
    import self_attention_for_hw

class MDGAN_S1_G(nn.Module):
	def __init__(self,	ngf=32, sn=False):
		super(MDGAN_S1_G, self).__init__()
		# input is 3 x 32 x 128 x 128  (duplicated by 3 x 1 x 128 x 128)
		self.relu = nn.ReLU(inplace = True)
		self.attn_1 = self_attention_for_hw.SelfAttentionForHW(ngf, self.relu)
		self.attn_2 = self_attention_for_hw.SelfAttentionForHW(ngf *2, self.relu)
		self.attn_3 = self_attention_for_hw.SelfAttentionForHW(ngf *4, self.relu)
		self.attn_4 = self_attention_for_hw.SelfAttentionForHW(ngf *8, self.relu)
		self.attn_5 = self_attention_for_hw.SelfAttentionForHW(ngf*16, self.relu)

		if sn:
			print("Use spectral normalization in S1 G.")
			
			self.downConv1 = spectral_norm(nn.Conv3d(3, ngf,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1),bias=False))
			self.downConv2 = spectral_norm(nn.Conv3d(ngf, ngf *2 ,(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),bias=False))
			self.downConv3 = spectral_norm(nn.Conv3d(ngf *2, ngf * 4, 4, 2, 1, bias=False))
			self.downConv4 = spectral_norm(nn.Conv3d(ngf * 4, ngf * 8, 4, 2, 1, bias=False))
			self.downConv5 = spectral_norm(nn.Conv3d(ngf * 8, ngf * 8, (2,4,4), stride=(1,1,1), padding=(0,0,0), bias=False))
			# self.downConv6 = spectral_norm(nn.Conv3d(ngf * 16, ngf * 16, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False))
			# get 
			self.downBN2 = nn.BatchNorm3d(ngf * 2)
			self.downBN3 = nn.BatchNorm3d(ngf * 4)
			self.downBN4 = nn.BatchNorm3d(ngf * 8)
			# self.downBN5 = nn.BatchNorm3d(ngf * 16)
			
			# self.upConv1 = spectral_norm(nn.ConvTranspose3d(ngf * 16, ngf * 16, (2,4,4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False ))
			# self.upConv2 = spectral_norm(nn.ConvTranspose3d(ngf * 16, ngf * 8, 4, 2, 1, bias=False))
			self.upConv2 = spectral_norm(nn.ConvTranspose3d(ngf * 8, ngf * 8, (2,4,4), stride=(1,1,1), padding=(0,0,0), bias=False))
			self.upConv3 = spectral_norm(nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False))
			self.upConv4 = spectral_norm(nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False))
			self.upConv5 = spectral_norm(nn.ConvTranspose3d(ngf * 2, ngf * 1, (4,4,4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False))
			self.upConv6 = spectral_norm(nn.ConvTranspose3d(ngf * 1, 3, (3,4,4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False))
			self.tanh = nn.Tanh()
			# self.upBN1 = nn.BatchNorm3d(ngf * 16)
			self.upBN2 = nn.BatchNorm3d(ngf * 8)
			self.upBN3 = nn.BatchNorm3d(ngf * 4)
			self.upBN4 = nn.BatchNorm3d(ngf * 2)
			self.upBN5 = nn.BatchNorm3d(ngf * 1)
			self.lrelu = nn.LeakyReLU(0.2, inplace=True)
		else:
			self.downConv1 = nn.Conv3d(3, ngf,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1),bias=False)
			self.downConv2 = nn.Conv3d(ngf, ngf *2 ,(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),bias=False)
			self.downConv3 = nn.Conv3d(ngf *2, ngf * 4, 4, 2, 1, bias=False)
			self.downConv4 = nn.Conv3d(ngf * 4, ngf * 8, 4, 2, 1, bias=False)
			self.downConv5 = nn.Conv3d(ngf * 8, ngf * 8, (2,4,4), stride=(1,1,1), padding=(0,0,0), bias=False)
			# self.downConv5 = nn.Conv3d(ngf * 8, ngf * 16, 4, 2, 1, bias=False)
			# self.downConv6 = nn.Conv3d(ngf * 16, ngf * 16, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
			# get 
			self.downBN2 = nn.BatchNorm3d(ngf * 2)
			self.downBN3 = nn.BatchNorm3d(ngf * 4)
			self.downBN4 = nn.BatchNorm3d(ngf * 8)
			# self.downBN5 = nn.BatchNorm3d(ngf * 16)
			self.relu = nn.ReLU(inplace = True)
			
			# self.upConv1 = nn.ConvTranspose3d(ngf * 16, ngf * 16, (2,4,4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False )
			# self.upConv2 = nn.ConvTranspose3d(ngf * 16, ngf * 8, 4, 2, 1, bias=False)
			self.upConv2 = nn.ConvTranspose3d(ngf * 8, ngf * 8, (2,4,4), stride=(1,1,1), padding=(0,0,0), bias=False)
			self.upConv3 = nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
			self.upConv4 = nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
			self.upConv5 = nn.ConvTranspose3d(ngf * 2, ngf * 1, (4,4,4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
			self.upConv6 = nn.ConvTranspose3d(ngf * 1, 3, (3,4,4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False)
			self.tanh = nn.Tanh()
			# self.upBN1 = nn.BatchNorm3d(ngf * 16)
			self.upBN2 = nn.BatchNorm3d(ngf * 8)
			self.upBN3 = nn.BatchNorm3d(ngf * 4)
			self.upBN4 = nn.BatchNorm3d(ngf * 2)
			self.upBN5 = nn.BatchNorm3d(ngf * 1)
			self.lrelu = nn.LeakyReLU(0.2, inplace=True)
	def forward(self, x):
		downx1 = self.downConv1(x)
		downx2 = self.downConv2(downx1)
		downx2 = self.downBN2(downx2)
		downx2 = self.lrelu(downx2)
		downx3 = self.downConv3(downx2)
		downx3 = self.downBN3(downx3)
		downx3 = self.lrelu(downx3)
		downx4 = self.downConv4(downx3)
		downx4 = self.downBN4(downx4)
		downx4 = self.lrelu(downx4)
		downx5 = self.downConv5(downx4)
		# downx5 = self.downBN5(downx5)
		# downx5 = self.lrelu(downx5)
		# downx6 = self.downConv6(downx5)
		
		# upx1 = self.upConv1(downx6)
		# upx1 = self.upBN1(upx1)
		# upx1 = self.relu(upx1)
		# upx1 = self.attn_5(downx5)[0] + upx1
		
		# upx2 = self.upConv2(upx1)
		upx2 = self.upConv2(downx5)
		upx2 = self.upBN2(upx2)
		upx2 = self.relu(upx2)
		upx2 = self.attn_4(downx4)[0] + upx2
		
		upx3 = self.upConv3(upx2)
		upx3 = self.upBN3(upx3)
		upx3 = self.relu(upx3)
		upx3 = self.attn_3(downx3)[0] + upx3

		upx4 = self.upConv4(upx3)
		upx4 = self.upBN4(upx4)
		upx4 = self.relu(upx4)
		upx4 = self.attn_2(downx2)[0] + upx4

		upx5 = self.upConv5(upx4)
		upx5 = self.upBN5(upx5)
		upx5 = self.relu(upx5)
		upx5 = self.attn_1(downx1)[0] + upx5

		upx6 = self.upConv6(upx5)
		upx6 = self.tanh(upx6)
		
		return upx6

class MDGAN_S2_G(nn.Module):
	def __init__(self, ngf=32, sn=False):
		super(MDGAN_S2_G, self).__init__()
		# input is 3 x 32 x 128 x 128  (duplicated by 3 x 1 x 128 x 128)

		self.relu = nn.ReLU(inplace = True)
		self.lrelu = nn.LeakyReLU(0.2, inplace=True)

		self.attn_1 = self_attention_for_hw.SelfAttentionForHW(ngf, self.relu)
		self.attn_2 = self_attention_for_hw.SelfAttentionForHW(ngf *2, self.relu)
		self.attn_3 = self_attention_for_hw.SelfAttentionForHW(ngf *4, self.relu)
		self.attn_4 = self_attention_for_hw.SelfAttentionForHW(ngf *8, self.relu)

		self.up_attn_1 = self_attention_for_hw.SelfAttentionForHW(ngf, self.lrelu)
		self.up_attn_2 = self_attention_for_hw.SelfAttentionForHW(ngf *2, self.lrelu)
		self.up_attn_3 = self_attention_for_hw.SelfAttentionForHW(ngf *4, self.lrelu)
		self.up_attn_4 = self_attention_for_hw.SelfAttentionForHW(ngf *8, self.lrelu)


		if sn:
			print("Use spectral normalization in S2 G.")
			self.downConv1 = spectral_norm(nn.Conv3d(3, ngf,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1),bias=False))
			self.downConv2 = spectral_norm(nn.Conv3d(ngf, ngf *2 ,(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),bias=False))
			self.downConv3 = spectral_norm(nn.Conv3d(ngf *2, ngf * 4, 4, 2, 1, bias=False))
			self.downConv4 = spectral_norm(nn.Conv3d(ngf * 4, ngf * 8, 4, 2, 1, bias=False))
			self.downConv5 = spectral_norm(nn.Conv3d(ngf * 8, ngf * 8, 4, 2, 1, bias=False))
			# self.downConv5 = spectral_norm(nn.Conv3d(ngf * 8, ngf * 16, 4, 2, 1, bias=False))
			# self.downConv6 = spectral_norm(nn.Conv3d(ngf * 16, ngf * 16, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False))
			# get 
			self.downBN2 = nn.BatchNorm3d(ngf * 2)
			self.downBN3 = nn.BatchNorm3d(ngf * 4)
			self.downBN4 = nn.BatchNorm3d(ngf * 8)
			# self.downBN5 = nn.BatchNorm3d(ngf * 16)
			self.relu = nn.ReLU(inplace = True)
			
			# self.upConv1 = spectral_norm(nn.ConvTranspose3d(ngf * 16, ngf * 16, (2,4,4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False ))
			# self.upConv2 = spectral_norm(nn.ConvTranspose3d(ngf * 16, ngf * 8, 4, 2, 1, bias=False))
			self.upConv2 = spectral_norm(nn.ConvTranspose3d(ngf * 8, ngf * 8, 4, 2, 1, bias=False))
			self.upConv3 = spectral_norm(nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False))
			self.upConv4 = spectral_norm(nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False))
			self.upConv5 = spectral_norm(nn.ConvTranspose3d(ngf * 2, ngf * 1, (4,4,4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False))
			self.upConv6 = spectral_norm(nn.ConvTranspose3d(ngf * 1, 3, (3,4,4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False))
			self.tanh = nn.Tanh()
			# self.upBN1 = nn.BatchNorm3d(ngf * 16)
			self.upBN2 = nn.BatchNorm3d(ngf * 8)
			self.upBN3 = nn.BatchNorm3d(ngf * 4)
			self.upBN4 = nn.BatchNorm3d(ngf * 2)
			self.upBN5 = nn.BatchNorm3d(ngf * 1)
			self.lrelu = nn.LeakyReLU(0.2, inplace=True)
		else:
			self.downConv1 = nn.Conv3d(3, ngf,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1),bias=False)
			self.downConv2 = nn.Conv3d(ngf, ngf *2 ,(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),bias=False)
			self.downConv3 = nn.Conv3d(ngf *2, ngf * 4, 4, 2, 1, bias=False)
			self.downConv4 = nn.Conv3d(ngf * 4, ngf * 8, 4, 2, 1, bias=False)
			self.downConv5 = nn.Conv3d(ngf * 8, ngf * 8, 4, 2, 1, bias=False)
			# self.downConv5 = nn.Conv3d(ngf * 8, ngf * 16, 4, 2, 1, bias=False)
			# self.downConv6 = nn.Conv3d(ngf * 16, ngf * 16, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
			# get 
			self.downBN2 = nn.BatchNorm3d(ngf * 2)
			self.downBN3 = nn.BatchNorm3d(ngf * 4)
			self.downBN4 = nn.BatchNorm3d(ngf * 8)
			# self.downBN5 = nn.BatchNorm3d(ngf * 16)
			
			
			# self.upConv1 = nn.ConvTranspose3d(ngf * 16, ngf * 16, (2,4,4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False )
			# self.upConv2 = nn.ConvTranspose3d(ngf * 16, ngf * 8, 4, 2, 1, bias=False)
			self.upConv2 = nn.ConvTranspose3d(ngf * 8, ngf * 8, 4, 2, 1, bias=False)
			self.upConv3 = nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
			self.upConv4 = nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
			self.upConv5 = nn.ConvTranspose3d(ngf * 2, ngf * 1, (4,4,4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
			self.upConv6 = nn.ConvTranspose3d(ngf * 1, 3, (3,4,4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False)
			self.tanh = nn.Tanh()
			# self.upBN1 = nn.BatchNorm3d(ngf * 16)
			self.upBN2 = nn.BatchNorm3d(ngf * 8)
			self.upBN3 = nn.BatchNorm3d(ngf * 4)
			self.upBN4 = nn.BatchNorm3d(ngf * 2)
			self.upBN5 = nn.BatchNorm3d(ngf * 1)
			

	def forward(self, x):
		# print('x.size(): ', x.size()) # -> [1, 3, 32, 64, 64]
		downx1 = self.downConv1(x)
		downx1 = self.attn_1(downx1)[0]
		# print('downx1.size(): ', downx1.size()) # -> [1, 32, 32, 32, 32]) 
		downx2 = self.downConv2(downx1)
		downx2 = self.downBN2(downx2)
		downx2 = self.lrelu(downx2)
		downx2 = self.attn_2(downx2)[0]
		# print('downx2.size(): ', downx2.size()) # -> [1, 64, 16, 16, 16]
		downx3 = self.downConv3(downx2)
		downx3 = self.downBN3(downx3)
		downx3 = self.lrelu(downx3)
		downx3 = self.attn_3(downx3)[0]
		# print('downx3.size(): ', downx3.size()) # -> [1, 128, 8, 8, 8]
		downx4 = self.downConv4(downx3)
		downx4 = self.downBN4(downx4)
		downx4 = self.lrelu(downx4)
		downx4 = self.attn_4(downx4)[0]
		# print('downx4.size(): ', downx4.size())  # -> [1, 256, 4, 4, 4]
		downx5 = self.downConv5(downx4)
		# print('downx5.size(): ', downx5.size()) # -> [1, 256, 2, 2, 2]
		# downx5 = self.downBN5(downx5)
		# downx5 = self.lrelu(downx5)
		# downx6 = self.downConv6(downx5)
		
		# upx1 = self.upConv1(downx6)
		# upx1 = self.upBN1(upx1)
		# upx1 = self.relu(upx1)
		# upx1 = downx5 + upx1
		
		upx2 = self.upConv2(downx5)
		upx2 = self.upBN2(upx2)
		upx2 = self.relu(upx2)
		upx2 = self.up_attn_4(upx2)[0]
		# print('upx2.size(): ', upx2.size()) # -> [1, 256, 4, 4, 4]
		upx2 = downx4 + upx2
		# print('upx2.size(): ', upx2.size()) # -> [1, 256, 2, 2, 2]
		
		upx3 = self.upConv3(upx2)
		upx3 = self.upBN3(upx3)
		upx3 = self.relu(upx3)
		upx3 = self.up_attn_3(upx3)[0]
		# print('upx3.size(): ', upx3.size()) # -> [1, 128, 4, 4, 4]
		upx3 = downx3 + upx3
		# print('upx3.size(): ', upx3.size())

		upx4 = self.upConv4(upx3)
		upx4 = self.upBN4(upx4)
		upx4 = self.relu(upx4)
		upx4 = self.up_attn_2(upx4)[0]
		# print('upx4.size(): ', upx4.size())
		#upx4 = downx2 + upx4

		upx5 = self.upConv5(upx4)
		upx5 = self.upBN5(upx5)
		upx5 = self.relu(upx5)
		upx5 = self.up_attn_1(upx5)[0]
		#upx5 = downx1 + upx5
		# print('upx5.size(): ', upx5.size())

		upx6 = self.upConv6(upx5)
		upx6 = self.tanh(upx6)
		# print('upx6.size(): ', upx6.size())
		
		return upx6

class MDGAN_S2_D(nn.Module):
	def __init__(self, ndf=32, sn=False, attn_loss=True):
		super(MDGAN_S2_D, self).__init__()
		self.attn_loss = attn_loss

		if attn_loss:
			self.lrelu = nn.LeakyReLU(0.2, inplace=True)
			self.attn_1 = self_attention_for_hw.SelfAttentionForHW(ndf, self.lrelu)
			self.attn_2 = self_attention_for_hw.SelfAttentionForHW(ndf *4, self.lrelu)

		if sn:
			print("Use spectral normalization in D.")
			self.slice1 = nn.Sequential(
				# input is 3 x 32 x 256 x 256
				spectral_norm(nn.Conv3d(3, ndf,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1),bias=False)),
				nn.LeakyReLU(0.2, inplace=True), 
			)
			self.slice2 = nn.Sequential(
				# ndf x 32 x 64 x 64 
				spectral_norm(nn.Conv3d(ndf, ndf *2 ,(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),bias=False)),
				nn.BatchNorm3d(ndf * 2),
				nn.LeakyReLU(0.2, inplace=True),			
				# (ndf*2) x 16 x 32 x 32
				spectral_norm(nn.Conv3d(ndf *2, ndf * 4, 4, 2, 1, bias=False)),
				nn.BatchNorm3d(ndf * 4),
				nn.LeakyReLU(0.2, inplace=True),
			)
			self.slice3 = nn.Sequential(
				#  (ndf*4) x 8 x 16 x 16
				spectral_norm(nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
				nn.BatchNorm3d(ndf * 8),
				nn.LeakyReLU(0.2, inplace=True),
				# # (ndf*8) x 4 x 8 x 8
				# spectral_norm(nn.Conv3d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)),
				# nn.BatchNorm3d(ndf * 16),
				# nn.LeakyReLU(inplace=True),
			)
			
			self.slice4 = nn.Sequential( 
				# (ndf*16) x 2 x 4 x 4
				spectral_norm(nn.Conv3d(ndf * 8, 1, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)),
				nn.Sigmoid(),
			)
		else:
			self.slice1 = nn.Sequential(
				# input is 3 x 32 x 256 x 256
				nn.Conv3d(3, ndf,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1),bias=False),
				nn.LeakyReLU(0.2, inplace=True), 
			)
			self.slice2 = nn.Sequential(
				# ndf x 32 x 64 x 64 
				nn.Conv3d(ndf, ndf *2 ,(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),bias=False),
				nn.BatchNorm3d(ndf * 2),
				nn.LeakyReLU(0.2, inplace=True),			
				# (ndf*2) x 16 x 32 x 32
				nn.Conv3d(ndf *2, ndf * 4, 4, 2, 1, bias=False),
				nn.BatchNorm3d(ndf * 4),
				nn.LeakyReLU(0.2, inplace=True),
			)
			self.slice3 = nn.Sequential(
				#  (ndf*4) x 8 x 16 x 16
				nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
				nn.BatchNorm3d(ndf * 8),
				nn.LeakyReLU(0.2, inplace=True),
				# # (ndf*8) x 4 x 8 x 8
				# nn.Conv3d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
				# nn.BatchNorm3d(ndf * 16),
				# nn.LeakyReLU(inplace=True),
			)
			self.slice4 = nn.Sequential( 
				# (ndf*16) x 2 x 4 x 4
				nn.Conv3d(ndf * 8, 1, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
				nn.Sigmoid(),
			)


	def forward(self, x):

		x1 = self.slice1(x)
		x2 = self.slice2(x1)
		x3 = self.slice3(x2)
		x4 = self.slice4(x3)

		if self.attn_loss:
			_, x1_attn = self.attn_1(x1)
			_, x2_attn = self.attn_2(x2)
			return x4.view(-1, 1), [x2_attn, x1_attn]
		else:
			return x4.view(-1, 1), [x2, x1]

				




class Sampler(object):
	"""Base class for all Samplers.

	Every Sampler subclass has to provide an __iter__ method, providing a way
	to iterate over indices of dataset elements, and a __len__ method that
	returns the length of the returned iterators.
	"""

	def __init__(self, data_source):
		pass

	def __iter__(self):
		raise NotImplementedError

	def __len__(self):
		raise NotImplementedError



class SubsetRandomSampler(Sampler):
	"""Samples elements randomly from a given list of indices, without replacement.

	Arguments:
		indices (list): a list of indices
	"""

	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return (self.indices[i] for i in self.indices)

	def __len__(self):
		return len(self.indices)


