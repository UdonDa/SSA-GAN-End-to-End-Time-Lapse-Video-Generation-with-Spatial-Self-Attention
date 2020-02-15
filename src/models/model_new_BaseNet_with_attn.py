import torch
import torch.nn as nn
from sys import exit
from torch.nn.utils import spectral_norm
try:
    from . import utils
except:
    import utils

try:
    from . import residual_block
except:
    import residual_block

try:
    from . import self_attention_for_hw
except:
    import self_attention_for_hw


class BaseNetGenerator(nn.Module):
    """
    encoder - resblock - decoderネットワーク.
    Conv, Deconvの後，空間方向のAttention計算を追加している(See self_attention_for_hw.py).
    ネットワーク図参考: https://www.researchgate.net/publication/328109215_DAU-GAN_Unsupervised_Object_Transfiguration_via_Deep_Attention_Unit_9th_International_Conference_BICS_2018_Xi'an_China_July_7-8_2018_Proceedings
    """
    def __init__(self, in_dim=3, out_dim=3, filters=32, norm='instance'):
        super(BaseNetGenerator, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.filters = filters
        self.relu = nn.ReLU() #TODO slopeの値の確認
        self.L_relu = nn.LeakyReLU(0.2, inplace=True)
        self.use_spectral_norm = True
        self.norm = norm

        self.conv_1 = utils.conv_first(self.in_dim, self.filters, self.relu, self.use_spectral_norm) # 32
        self.conv_2 = utils.conv_421(self.filters, self.filters*2, self.relu, self.use_spectral_norm) # 64
        self.attn_Ehw_2 = self_attention_for_hw.SelfAttentionForHW(self.filters*2, self.relu)
        self.conv_3 = utils.conv_421(self.filters*2, self.filters*4, self.relu, self.use_spectral_norm) # 128
        self.attn_Ehw_3 = self_attention_for_hw.SelfAttentionForHW(self.filters*4, self.relu)
        self.conv_4 = utils.conv_421(self.filters*4, self.filters*8, self.relu, self.use_spectral_norm) # 256
        self.attn_Ehw_4 = self_attention_for_hw.SelfAttentionForHW(self.filters*8, self.relu)
        self.conv_5 = utils.conv_421(self.filters*8, self.filters*16, self.relu, self.use_spectral_norm) # 512
        self.attn_Ehw_5 = self_attention_for_hw.SelfAttentionForHW(self.filters*16, self.relu)

        self.conv_6 = utils.conv_6(self.filters*16, self.filters*16, self.relu, self.use_spectral_norm) # 512
        self.attn_Ehw_6 = self_attention_for_hw.SelfAttentionForHW(self.filters*16, self.relu)

        residual_blocks = []
        for _ in range(6):
          residual_blocks.append(residual_block.ResidualBlock(self.filters*16))
        self.residual_blocks = nn.Sequential(*residual_blocks)

        # self.deconv_1 = utils.deconv_first(self.filters*16, self.filters*8, self.use_spectral_norm) # 512 -> 256
        self.deconv_1 = utils.deconv_421(self.filters*16, self.filters*16, self.L_relu, self.use_spectral_norm)
        self.attn_Dhw_1 = self_attention_for_hw.SelfAttentionForHW(self.filters*16, self.L_relu)

        self.deconv_2 = utils.deconv_421(self.filters*16, self.filters*8, self.L_relu, self.use_spectral_norm) # 256 -> 128
        self.attn_Dhw_2 = self_attention_for_hw.SelfAttentionForHW(self.filters*8, self.L_relu)
        self.deconv_3 = utils.deconv_421(self.filters*8, self.filters*4, self.L_relu, self.use_spectral_norm) # 128 -> 64
        self.attn_Dhw_3 = self_attention_for_hw.SelfAttentionForHW(self.filters*4, self.L_relu)
        self.deconv_4 = utils.deconv_421(self.filters*4, self.filters*2, self.L_relu, self.use_spectral_norm) # 64 -> 32
        self.attn_Dhw_4 = self_attention_for_hw.SelfAttentionForHW(self.filters*2, self.L_relu)
        
        self.deconv_5 = utils.deconv_421(self.filters*2, self.filters, self.L_relu, self.use_spectral_norm) # 128 -> 64
        self.attn_Dhw_5 = self_attention_for_hw.SelfAttentionForHW(self.filters, self.L_relu)

        if self.use_spectral_norm:
            self.out_conv = spectral_norm(nn.ConvTranspose3d(self.filters, self.out_dim, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)))
        else:
            self.out_conv = nn.ConvTranspose3d(self.filters, self.out_dim, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1))

        self.tanh = nn.Tanh()
        



    def forward(self, x):
        # print("G: Down sampling")
        # utils.show_size(x)
        f1 = self.conv_1(x) # -> [1, 32, 32, 128, 128]
        # utils.show_size(f1)
        

        f2 = self.conv_2(f1) # -> [1, 64, 16, 64, 64]
        out, _ = self.attn_Ehw_2(f2)
        # utils.show_size(out)
        

        f3 = self.conv_3(out) # -> [1, 128, 8, 32, 32]
        out, _ = self.attn_Ehw_3(f3)
        # utils.show_size(out)

        f4 = self.conv_4(out)
        out, _ = self.attn_Ehw_4(f4)
        # utils.show_size(out)

        f5 = self.conv_5(out)
        out, _ = self.attn_Ehw_5(f5)
        # utils.show_size(out)

        f6 = self.conv_6(f5)
        out, _ = self.attn_Ehw_6(f6)
        # utils.show_size(out)

    
        f6 = self.residual_blocks(out)
        # utils.show_size(f6)

        f7 = self.deconv_1(f6)
        out, _ = self.attn_Dhw_1(f7)
        # utils.show_size(out)
        

        f8 = self.deconv_2(out)
        out, _ = self.attn_Dhw_2(f8)
        # utils.show_size(out)
        

        f9 = self.deconv_3(out)
        out, _ = self.attn_Dhw_3(f9)
        # utils.show_size(out)

        f10 = self.deconv_4(out)
        out, _ = self.attn_Dhw_4(f10)
        # utils.show_size(out)

        f11 = self.deconv_5(out)
        out, _ = self.attn_Dhw_5(f11)
        # utils.show_size(out)
        

        out_conv = self.out_conv(out)
        out = self.tanh(out_conv)
        return out

class BaseNetDiscriminator(nn.Module):
    """
        BaseNetDiscriminator
    """
    def __init__(self):
        super(BaseNetDiscriminator, self).__init__()
        self.in_dim = 3
        self.out_dim = 1
        self.filters = 32
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.use_spectral_norm = True

        layers = []
        if self.use_spectral_norm:
            layers.append(spectral_norm(nn.Conv3d(3, self.filters, kernel_size=(4,4,4), stride=(1,1,1), padding=0)))

            curr_dim = self.filters
            for _ in range(1, 5):
                layers.append(spectral_norm(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))),
                layers.append(nn.BatchNorm3d(curr_dim*2)),
                layers.append(nn.LeakyReLU(0.2))
                curr_dim = curr_dim * 2

            self.main = nn.Sequential(*layers)
            # self.conv1 = spectral_norm(nn.Conv3d(curr_dim, 1, kernel_size=(3,4,4), stride=(1,2,2), padding=1, bias=False))
            self.conv1 = spectral_norm(nn.Conv3d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            layers.append(nn.Conv3d(3, self.filters, kernel_size=(3,4,4), stride=(1,2,2), padding=1))

            curr_dim = self.filters
            for _ in range(1, 5):
                layers.append(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
                layers.append(nn.LeakyReLU(0.01))
                curr_dim = curr_dim * 2

            self.main = nn.Sequential(*layers)
            # self.conv1 = nn.Conv3d(curr_dim, 1, kernel_size=(3,4,4), stride=(1,2,2), padding=1, bias=False)
            self.conv1 = nn.Conv3d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        out = self.main(x)
        out = self.conv1(out)
        return out.view(-1)


# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     image_size = 128
#     T = 32 # times of duplicated
#     x = torch.Tensor(1, 3, T, image_size, image_size)
#     x.to(device)
#     print("x size: {}".format(x.size()))

#     baseNetG = BaseNetGenerator('', in_dim=3, out_dim=3, filters=32)
#     baseNetD = BaseNetDiscriminator()

#     out = baseNetG(x)
#     print("BaseNetGenerator output size: {}".format(out.size()))
#     out = baseNetD(x)
#     print("BaseNetDiscriminator output size: {}".format(out.size()))