import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
try:
    from . import self_attention_for_thw
    from . import self_attention_for_thw_no_gamma
except:
    import self_attention_for_thw
    import self_attention_for_thw_no_gamma

class MDGAN_S1_G(nn.Module):
    def __init__(self, ngf=32, sn=False, args=None):
        super(MDGAN_S1_G, self).__init__()
        # input is 3 x 32 x 128 x 128  (duplicated by 3 x 1 x 128 x 128)
        """parameter"""

        self.relu = nn.ReLU(inplace = True)
        if args.attention_type == "THW":
            print("Use THW.")
            SA = self_attention_for_thw.SelfAttentionBlock
        elif args.attention_type == "noGammaTHW":
            print("Use noGammaTHW.")
            SA = self_attention_for_thw_no_gamma.SelfAttentionBlock

        # self.attn_1 = SA(ngf, self.relu, c_cal=args.c_cal)
        # self.attn_2 = SA(ngf *2, self.relu, c_cal=args.c_cal)
        # self.attn_3 = SA(ngf *4, self.relu, c_cal=args.c_cal)
        # self.attn_4 = SA(ngf *8, self.relu, c_cal=args.c_cal)
        # self.attn_5 = SA(ngf*16, self.relu, c_cal=args.c_cal)

        self.up_attn_1 = SA(ngf, self.relu, c_cal=args.c_cal)
        # self.up_attn_2 = SA(ngf *2, self.relu, c_cal=args.c_cal)
        # self.up_attn_3 = SA(ngf *4, self.relu, c_cal=args.c_cal)
        # self.up_attn_4 = SA(ngf *8, self.relu, c_cal=args.c_cal)
        # self.up_attn_5 = SA(ngf*16, self.relu, c_cal=args.c_cal)

        if sn:
            print("Use spectral normalization in S1 G.")
            
            self.downConv1 = spectral_norm(nn.Conv3d(3, ngf,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1),bias=False))
            self.downConv2 = spectral_norm(nn.Conv3d(ngf, ngf *2 ,(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),bias=False))
            self.downConv3 = spectral_norm(nn.Conv3d(ngf *2, ngf * 4, 4, 2, 1, bias=False))
            self.downConv4 = spectral_norm(nn.Conv3d(ngf * 4, ngf * 8, 4, 2, 1, bias=False))
            self.downConv5 = spectral_norm(nn.Conv3d(ngf * 8, ngf * 16, 4, 2, 1, bias=False))
            self.downConv6 = spectral_norm(nn.Conv3d(ngf * 16, ngf * 16, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False))
            # get 
            self.downBN2 = nn.BatchNorm3d(ngf * 2)
            self.downBN3 = nn.BatchNorm3d(ngf * 4)
            self.downBN4 = nn.BatchNorm3d(ngf * 8)
            self.downBN5 = nn.BatchNorm3d(ngf * 16)
            
            self.upConv1 = spectral_norm(nn.ConvTranspose3d(ngf * 16, ngf * 16, (2,4,4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False ))
            self.upConv2 = spectral_norm(nn.ConvTranspose3d(ngf * 16 * 2, ngf * 8, 4, 2, 1, bias=False))
            self.upConv3 = spectral_norm(nn.ConvTranspose3d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=False))
            self.upConv4 = spectral_norm(nn.ConvTranspose3d(ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=False))
            self.upConv5 = spectral_norm(nn.ConvTranspose3d(ngf * 2 * 2, ngf * 1, (4,4,4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False))
            self.upConv6 = spectral_norm(nn.ConvTranspose3d(ngf * 1 * 2, 3, (3,4,4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False))
            self.tanh = nn.Tanh()
            self.upBN1 = nn.BatchNorm3d(ngf * 16)
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
            self.downConv5 = nn.Conv3d(ngf * 8, ngf * 16, 4, 2, 1, bias=False)
            self.downConv6 = nn.Conv3d(ngf * 16, ngf * 16, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
            # get 
            self.downBN2 = nn.BatchNorm3d(ngf * 2)
            self.downBN3 = nn.BatchNorm3d(ngf * 4)
            self.downBN4 = nn.BatchNorm3d(ngf * 8)
            self.downBN5 = nn.BatchNorm3d(ngf * 16)
            self.relu = nn.ReLU(inplace = True)
            
            self.upConv1 = nn.ConvTranspose3d(ngf * 16, ngf * 16, (2,4,4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False )
            self.upConv2 = nn.ConvTranspose3d(ngf * 16 * 2, ngf * 8, 4, 2, 1, bias=False)
            self.upConv3 = nn.ConvTranspose3d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=False)
            self.upConv4 = nn.ConvTranspose3d(ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=False)
            self.upConv5 = nn.ConvTranspose3d(ngf * 2 * 2, ngf * 1, (4,4,4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
            self.upConv6 = nn.ConvTranspose3d(ngf * 1 * 2, 3, (3,4,4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False)
            self.tanh = nn.Tanh()
            self.upBN1 = nn.BatchNorm3d(ngf * 16)
            self.upBN2 = nn.BatchNorm3d(ngf * 8)
            self.upBN3 = nn.BatchNorm3d(ngf * 4)
            self.upBN4 = nn.BatchNorm3d(ngf * 2)
            self.upBN5 = nn.BatchNorm3d(ngf * 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        # print(x.size())
        downx1 = self.downConv1(x)
        # print(downx1.size())
        # downx1 = self.attn_1(downx1)[0]

        downx2 = self.downConv2(downx1)
        downx2 = self.downBN2(downx2)
        downx2 = self.lrelu(downx2)
        # downx2 = self.attn_2(downx2)[0]

        downx3 = self.downConv3(downx2)
        downx3 = self.downBN3(downx3)
        downx3 = self.lrelu(downx3)
        # downx3 = self.attn_3(downx3)[0]

        downx4 = self.downConv4(downx3)
        downx4 = self.downBN4(downx4)
        downx4 = self.lrelu(downx4)
        # downx4 = self.attn_4(downx4)[0]

        downx5 = self.downConv5(downx4)
        downx5 = self.downBN5(downx5)
        downx5 = self.lrelu(downx5)
        # downx5 = self.attn_5(downx5)[0]

        downx6 = self.downConv6(downx5)
        
        upx1 = self.upConv1(downx6)
        upx1 = self.upBN1(upx1)
        upx1 = self.relu(upx1)
        # upx1 = self.up_attn_5(upx1)[0]
        # upx1 = downx5 + upx1
        upx1 = torch.cat((downx5, upx1), dim=1)
        
        upx2 = self.upConv2(upx1)
        upx2 = self.upBN2(upx2)
        upx2 = self.relu(upx2)
        # upx2 = self.up_attn_4(upx2)[0]
        # upx2 = downx4 + upx2
        upx2 = torch.cat((upx2, downx4), dim=1)
        
        upx3 = self.upConv3(upx2)
        upx3 = self.upBN3(upx3)
        upx3 = self.relu(upx3)
        # upx3 = self.up_attn_3(upx3)[0]
        # upx3 = downx3 + upx3
        upx3 = torch.cat((upx3, downx3), dim=1)

        upx4 = self.upConv4(upx3)
        upx4 = self.upBN4(upx4)
        upx4 = self.relu(upx4)
        # upx4 = self.up_attn_2(upx4)[0]
        # upx4 = downx2 + upx4
        upx4 = torch.cat((upx4, downx2), dim=1)

        upx5 = self.upConv5(upx4)
        upx5 = self.upBN5(upx5)
        upx5 = self.relu(upx5)
        upx5 = self.up_attn_1(upx5)[0]
        # upx5 = downx1 + upx5
        upx5 = torch.cat((upx5, downx1), dim=1)

        upx6 = self.upConv6(upx5)
        upx6 = self.tanh(upx6)
        
        return upx6

class MDGAN_S2_G(nn.Module):
    def __init__(self, ngf=32, sn=False):
        super(MDGAN_S2_G, self).__init__()
        # input is 3 x 32 x 128 x 128  (duplicated by 3 x 1 x 128 x 128)

        self.down_conv_attention = True
        self.up_conv_attention = True

        self.relu = nn.ReLU(inplace = True)
        self.attn_1 = self_attention_for_hw.SelfAttentionForHW(ngf, self.relu)
        self.attn_2 = self_attention_for_hw.SelfAttentionForHW(ngf *2, self.relu)
        self.attn_3 = self_attention_for_hw.SelfAttentionForHW(ngf *4, self.relu)
        self.attn_4 = self_attention_for_hw.SelfAttentionForHW(ngf *8, self.relu)
        self.attn_5 = self_attention_for_hw.SelfAttentionForHW(ngf*16, self.relu)

        self.up_attn_1 = self_attention_for_hw.SelfAttentionForHW(ngf, self.relu)
        self.up_attn_2 = self_attention_for_hw.SelfAttentionForHW(ngf *2, self.relu)
        self.up_attn_3 = self_attention_for_hw.SelfAttentionForHW(ngf *4, self.relu)
        self.up_attn_4 = self_attention_for_hw.SelfAttentionForHW(ngf *8, self.relu)
        self.up_attn_5 = self_attention_for_hw.SelfAttentionForHW(ngf*16, self.relu)

        if sn:
            print("Use spectral normalization in S2 G.")
            self.downConv1 = spectral_norm(nn.Conv3d(3, ngf,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1),bias=False))
            self.downConv2 = spectral_norm(nn.Conv3d(ngf, ngf *2 ,(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),bias=False))
            self.downConv3 = spectral_norm(nn.Conv3d(ngf *2, ngf * 4, 4, 2, 1, bias=False))
            self.downConv4 = spectral_norm(nn.Conv3d(ngf * 4, ngf * 8, 4, 2, 1, bias=False))
            self.downConv5 = spectral_norm(nn.Conv3d(ngf * 8, ngf * 16, 4, 2, 1, bias=False))
            self.downConv6 = spectral_norm(nn.Conv3d(ngf * 16, ngf * 16, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False))
            # get 
            self.downBN2 = nn.BatchNorm3d(ngf * 2)
            self.downBN3 = nn.BatchNorm3d(ngf * 4)
            self.downBN4 = nn.BatchNorm3d(ngf * 8)
            self.downBN5 = nn.BatchNorm3d(ngf * 16)
            self.relu = nn.ReLU(inplace = True)
            
            self.upConv1 = spectral_norm(nn.ConvTranspose3d(ngf * 16, ngf * 16, (2,4,4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False ))
            self.upConv2 = spectral_norm(nn.ConvTranspose3d(ngf * 16, ngf * 8, 4, 2, 1, bias=False))
            self.upConv3 = spectral_norm(nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False))
            self.upConv4 = spectral_norm(nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False))
            self.upConv5 = spectral_norm(nn.ConvTranspose3d(ngf * 2, ngf * 1, (4,4,4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False))
            self.upConv6 = spectral_norm(nn.ConvTranspose3d(ngf * 1, 3, (3,4,4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False))
            self.tanh = nn.Tanh()
            self.upBN1 = nn.BatchNorm3d(ngf * 16)
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
            self.downConv5 = nn.Conv3d(ngf * 8, ngf * 16, 4, 2, 1, bias=False)
            self.downConv6 = nn.Conv3d(ngf * 16, ngf * 16, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
            # get 
            self.downBN2 = nn.BatchNorm3d(ngf * 2)
            self.downBN3 = nn.BatchNorm3d(ngf * 4)
            self.downBN4 = nn.BatchNorm3d(ngf * 8)
            self.downBN5 = nn.BatchNorm3d(ngf * 16)
            self.relu = nn.ReLU(inplace = True)
            
            self.upConv1 = nn.ConvTranspose3d(ngf * 16, ngf * 16, (2,4,4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False )
            self.upConv2 = nn.ConvTranspose3d(ngf * 16, ngf * 8, 4, 2, 1, bias=False)
            self.upConv3 = nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
            self.upConv4 = nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
            self.upConv5 = nn.ConvTranspose3d(ngf * 2, ngf * 1, (4,4,4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
            self.upConv6 = nn.ConvTranspose3d(ngf * 1, 3, (3,4,4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False)
            self.tanh = nn.Tanh()
            self.upBN1 = nn.BatchNorm3d(ngf * 16)
            self.upBN2 = nn.BatchNorm3d(ngf * 8)
            self.upBN3 = nn.BatchNorm3d(ngf * 4)
            self.upBN4 = nn.BatchNorm3d(ngf * 2)
            self.upBN5 = nn.BatchNorm3d(ngf * 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        downx1 = self.downConv1(x)
        if self.down_conv_attention:
            downx1 = self.attn_1(downx1)[0]

        downx2 = self.downConv2(downx1)
        downx2 = self.downBN2(downx2)
        downx2 = self.lrelu(downx2)
        if self.down_conv_attention:
            downx2 = self.attn_2(downx2)[0]

        downx3 = self.downConv3(downx2)
        downx3 = self.downBN3(downx3)
        downx3 = self.lrelu(downx3)
        if self.down_conv_attention:
            downx3 = self.attn_3(downx3)[0]

        downx4 = self.downConv4(downx3)
        downx4 = self.downBN4(downx4)
        downx4 = self.lrelu(downx4)
        if self.down_conv_attention:
            downx4 = self.attn_4(downx4)[0]

        downx5 = self.downConv5(downx4)
        downx5 = self.downBN5(downx5)
        downx5 = self.lrelu(downx5)
        if self.down_conv_attention:
            downx5 = self.attn_5(downx5)[0]

        downx6 = self.downConv6(downx5)
        
        upx1 = self.upConv1(downx6)
        upx1 = self.upBN1(upx1)
        upx1 = self.relu(upx1)
        if self.up_conv_attention:
            upx1 = self.up_attn_5(upx1)[0]
        upx1 = downx5 + upx1
        
        upx2 = self.upConv2(upx1)
        upx2 = self.upBN2(upx2)
        upx2 = self.relu(upx2)
        if self.up_conv_attention:
            upx2 = self.up_attn_4(upx2)[0]
        upx2 = downx4 + upx2
        
        upx3 = self.upConv3(upx2)
        upx3 = self.upBN3(upx3)
        upx3 = self.relu(upx3)
        if self.up_conv_attention:
            upx3 = self.up_attn_3(upx3)[0]
        upx3 = downx3 + upx3

        upx4 = self.upConv4(upx3)
        upx4 = self.upBN4(upx4)
        upx4 = self.relu(upx4)
        if self.up_conv_attention:
            upx4 = self.up_attn_2(upx4)[0]
        #upx4 = downx2 + upx4

        upx5 = self.upConv5(upx4)
        upx5 = self.upBN5(upx5)
        upx5 = self.relu(upx5)
        if self.up_conv_attention:
            upx5 = self.up_attn_1(upx5)[0]
        #upx5 = downx1 + upx5

        upx6 = self.upConv6(upx5)
        upx6 = self.tanh(upx6)
        
        return upx6




class MDGAN_S2_D(nn.Module):
    def __init__(self, ndf=32, sn=False):
        super(MDGAN_S2_D, self).__init__()

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
                # (ndf*8) x 4 x 8 x 8
                spectral_norm(nn.Conv3d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)),
                nn.BatchNorm3d(ndf * 16),
                nn.LeakyReLU(inplace=True),
            )
            
            self.slice4 = nn.Sequential( 
                # (ndf*16) x 2 x 4 x 4
                spectral_norm(nn.Conv3d(ndf * 16, 1, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)),
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
                # (ndf*8) x 4 x 8 x 8
                nn.Conv3d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
                nn.BatchNorm3d(ndf * 16),
                nn.LeakyReLU(inplace=True),
            )
            self.slice4 = nn.Sequential( 
                # (ndf*16) x 2 x 4 x 4
                nn.Conv3d(ndf * 16, 1, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
                nn.Sigmoid(),
            )


    def forward(self, x):
        x1 = self.slice1(x)
        x2 = self.slice2(x1)
        x3 = self.slice3(x2)
        x4 = self.slice4(x3)

        return x4.view(-1, 1), [x2, x1]

if __name__ == "__main__":
    # D = MDGAN_S2_D(sn=True)
    # x = torch.randn(1,3,32,128,128)
    # y = D(x)

    G2 = MDGAN_S2_G(sn=True)
    G1 = MDGAN_S1_G(sn=True)

    x = torch.randn(1,3,32,128,128)
    y = G2(x)
    y = G1(x)




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


