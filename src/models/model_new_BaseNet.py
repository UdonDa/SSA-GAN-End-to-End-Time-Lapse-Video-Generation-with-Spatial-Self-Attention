"""
論文メモ
GとDの最初と最後のレイヤーにBNとactivationは入れないっぽい.
Gは最後Tanhする.
"""
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


class BaseNetGenerator(nn.Module):
    """
    encoder - resnet - decoderネットワーク.
    Attentionがない.
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
        self.conv_2 = utils.conv_421(self.filters*2, self.filters*4, self.activation, self.use_spectral_norm) # 128
        # self.conv_3 = utils.conv_421(self.filters*4, self.filters*8, self.activation) # 256

        residual_blocks = []
        for _ in range(6):
          # TODO: 3DResBlockを追加
          residual_blocks.append(residual_block.ResidualBlock(self.filters*4))
        self.residual_blocks = nn.Sequential(*residual_blocks)

        # self.deconv_1 = utils.deconv_421(self.filters*8, self.filters*4, self.activation)
        self.deconv_2 = utils.deconv_421(self.filters*4, self.filters*2, self.activation, self.use_spectral_norm)
        self.deconv_3 = utils.deconv_421(self.filters*2, self.filters, self.activation, self.use_spectral_norm)

        self.out_conv = spectral_norm(nn.Conv3d(self.filters, self.out_dim, kernel_size=7, stride=1, padding=3))
        self.tanh = nn.Tanh()


    def forward(self, x):
        # print("G: Down sampling")
        conv_first = self.conv_first(x) # -> [1, 32, 32, 128, 128]
        # utils.show_size(conv_first)
        conv_1 = self.conv_1(conv_first) # -> [1, 64, 16, 64, 64]
        # utils.show_size(conv_1)
        conv_2 = self.conv_2(conv_1) # -> [1, 128, 8, 32, 32]
        # utils.show_size(conv_2)
        # conv_3 = self.conv_3(conv_2) # -> 
        # utils.show_size(conv_3)

        resblocks = self.residual_blocks(conv_2)

        # print("G: Up sampling")
        # deconv_1 = self.deconv_1(conv_3) # -> [1, 512, 2, 4, 4]
        # utils.show_size(deconv_1)
        deconv_2 = self.deconv_2(resblocks) # -> [1, 64, 16, 64, 64]
        # utils.show_size(deconv_2)
        deconv_3 = self.deconv_3(deconv_2) # -> [1, 3, 32, 128, 128]
        # utils.show_size(deconv_3)
        out_conv = self.out_conv(deconv_3)
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
        self.filters = 64
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        layers = []
        layers.append(spectral_norm(nn.Conv3d(3, self.filters, kernel_size=(3,4,4), stride=(1,2,2), padding=1)))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = self.filters
        for _ in range(1, 5):
            layers.append(spectral_norm(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = spectral_norm(nn.Conv3d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        out = self.main(x)
        # return torch.sigmoid(self.conv1(out).view(-1, 1).squeeze(1))
        # return self.conv1(out).view(-1, 1).squeeze(1)
        return self.conv1(out)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 128
    T = 32 # times of duplicated
    x = torch.Tensor(1, 3, T, image_size, image_size)
    x.to(device)
    print("x size: {}".format(x.size()))

    baseNetG = BaseNetGenerator(in_dim=3, out_dim=3, filters=32)
    baseNetD = BaseNetDiscriminator()

    out = baseNetG(x)
    print("BaseNetGenerator output size: {}".format(out.size()))
    out = baseNetD(x)
    print("BaseNetDiscriminator output size: {}".format(out.size()))