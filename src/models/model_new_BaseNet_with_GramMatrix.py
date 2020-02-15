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
    def __init__(self, in_dim=3, out_dim=3, filters=64):
        super(BaseNetGenerator, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.filters = filters
        self.activation = nn.ReLU() #TODO slopeの値の確認
        self.use_spectral_norm = True

        self.conv_first = utils.conv_first_encdec(self.in_dim, self.filters, self.activation, self.use_spectral_norm) # 64

        self.conv_1 = utils.conv_421(self.filters, self.filters*2, self.activation, self.use_spectral_norm) # 64
        self.attn_hw_1 = self_attention_for_hw.SelfAttentionForHW(self.filters, self.activation)
        self.conv_2 = utils.conv_421(self.filters*2, self.filters*4, self.activation, self.use_spectral_norm) # 128
        self.attn_hw_2 = self_attention_for_hw.SelfAttentionForHW(self.filters*2, self.activation)

        residual_blocks = []
        for _ in range(6):
          residual_blocks.append(residual_block.ResidualBlock(self.filters*4))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        self.attn_hw_3 = self_attention_for_hw.SelfAttentionForHW(self.filters*4, self.activation)

        self.deconv_2 = utils.deconv_421(self.filters*4, self.filters*2, self.activation, self.use_spectral_norm)
        self.attn_hw_4 = self_attention_for_hw.SelfAttentionForHW(self.filters*2, self.activation)
        self.deconv_3 = utils.deconv_421(self.filters*2, self.filters, self.activation, self.use_spectral_norm)
        self.attn_hw_5 = self_attention_for_hw.SelfAttentionForHW(self.filters, self.activation)

        if self.use_spectral_norm:
            self.out_conv = spectral_norm(nn.Conv3d(self.filters, self.out_dim, kernel_size=7, stride=1, padding=3))
        else:
            self.out_conv = nn.Conv3d(self.filters, self.out_dim, kernel_size=7, stride=1, padding=3)

        self.tanh = nn.Tanh()


    def forward(self, x):
        # print("G: Down sampling")
        f1 = self.conv_first(x) # -> [1, 32, 32, 128, 128]
        # utils.show_size(conv_first)
        out, _ = self.attn_hw_1(f1) # -> [1, 32, 32, 128, 128], [1, 16384, 16384]
        # utils.show_size(out)
        # utils.show_size(_)
        f1_hat = f1 * out + f1

        f2 = self.conv_1(f1_hat) # -> [1, 64, 16, 64, 64]
        # utils.show_size(f2)
        out, _ = self.attn_hw_2(f2)
        f2_hat = f2 * out + f2

        f3 = self.conv_2(f2_hat) # -> [1, 128, 8, 32, 32]
        # utils.show_size(conv_2)

        f3 = self.residual_blocks(f3)
        # out, _ = self.attn_hw_3()
        # f3_hat = f3 * out + f3

        # print("G: Up sampling")
        f4 = self.deconv_2(f3) # -> [1, 64, 16, 64, 64]
        # utils.show_size(deconv_2)
        out, _ = self.attn_hw_4(f4)
        f4_hat = f4 * out + out

        f5 = self.deconv_3(f4_hat) # -> [1, 3, 32, 128, 128]
        # utils.show_size(f5)
        out, _ = self.attn_hw_5(f5)
        f5_hat = f5 * out + out
        
        out_conv = self.out_conv(f5_hat)
        out = self.tanh(out_conv)
        # return out, [f1, f2, f4, f5]
        return out, [f1, f2]

class BaseNetDiscriminator(nn.Module):
    """
        BaseNetDiscriminator
    """
    def __init__(self):
        super(BaseNetDiscriminator, self).__init__()
        self.in_dim = 3
        self.out_dim = 1
        self.filters = 64
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.use_spectral_norm = True

        self.conv_1 = utils.conv_first(self.in_dim, self.filters, self.use_spectral_norm) # 32
        self.conv_2 = utils.conv_421_D(self.filters, self.filters*2, self.activation, self.use_spectral_norm) # 64
        self.conv_3 = utils.conv_421_D(self.filters*2, self.filters*4, self.activation, self.use_spectral_norm) # 128
        self.conv_4 = utils.conv_421_D(self.filters*4, self.filters*8, self.activation, self.use_spectral_norm) # 256
        self.conv_5 = utils.conv_421_D(self.filters*8, self.filters*16, self.activation, self.use_spectral_norm) # 512
        self.conv_6 = nn.Conv3d(self.filters*16, self.out_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv_1 = self.conv_1(x) # -> [1, 32, 32, 64, 64]
        conv_2 = self.conv_2(conv_1) # -> [1, 64, 16, 32, 32]
        conv_3 = self.conv_3(conv_2) # -> [1, 128, 8, 16, 16]
        conv_4 = self.conv_4(conv_3) # -> [1, 256, 4, 8, 8]
        conv_5 = self.conv_5(conv_4) # -> [1, 512, 2, 4, 4]
        conv_6 = self.conv_6(conv_5) # -> [1, 1, 2, 4, 4]
        return conv_6, [conv_1, conv_2, conv_3, conv_4, conv_5]


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