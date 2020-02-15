import torch
import torch.nn as nn
try:
    from . import utils
except:
    import utils

class RefineNetGenerator(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, filters=32):
        super(RefineNetGenerator, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.filters = filters
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=False) #TODO slopeの値の確認

        self.conv_1 = utils.conv_first(self.in_dim, self.filters) # 32
        self.conv_2 = utils.conv_421(self.filters, self.filters*2, self.activation) # 64
        self.conv_3 = utils.conv_421(self.filters*2, self.filters*4, self.activation) # 128
        self.conv_4 = utils.conv_421(self.filters*4, self.filters*8, self.activation) # 256
        self.conv_5 = utils.conv_421(self.filters*8, self.filters*16, self.activation) # 512
        self.conv_6 = utils.conv_last(self.filters*16, self.filters*16) # 512
        
        self.activation = nn.ReLU()

        self.deconv_1 = utils.deconv_first(self.filters*16, self.filters*16)
        self.deconv_2 = utils.deconv_421(self.filters*32, self.filters*8, self.activation)
        self.deconv_3 = utils.deconv_421(self.filters*16, self.filters*4, self.activation)
        self.deconv_4 = utils.deconv_421(self.filters*8, self.filters*2, self.activation)
        self.deconv_5 = utils.deconv_421(self.filters*2, self.filters, self.activation)
        self.deconv_6 = utils.deconv_last(self.filters, self.out_dim)

    def forward(self, x):
        # print("G: Down sampling")
        conv_1 = self.conv_1(x) # -> [1, 32, 32, 64, 64]
        # utils.show_size(conv_1)
        conv_2 = self.conv_2(conv_1) # -> [1, 64, 16, 32, 32]
        # utils.show_size(conv_2)
        conv_3 = self.conv_3(conv_2) # -> [1, 128, 8, 16, 16]
        # utils.show_size(conv_3)
        conv_4 = self.conv_4(conv_3) # -> [1, 256, 4, 8, 8]
        # utils.show_size(conv_4)
        conv_5 = self.conv_5(conv_4) # -> [1, 512, 2, 4, 4]
        # utils.show_size(conv_5)
        conv_6 = self.conv_6(conv_5) # -> [1, 512, 1, 1, 1]
        # utils.show_size(conv_6)

        # print("G: Up sampling")
        deconv_1 = self.deconv_1(conv_6) # -> [1, 512, 2, 4, 4]
        # utils.show_size(deconv_1)
        concat_1 = torch.cat([conv_5, deconv_1], dim=1) # -> [1, 1024, 2, 4, 4]
        # utils.show_size(concat_1)

        deconv_2 = self.deconv_2(concat_1) # -> [1, 256, 4, 8, 8]
        # utils.show_size(deconv_2)
        concat_2 = torch.cat([conv_4, deconv_2], dim=1) # -> [1, 512, 4, 8, 8]
        # utils.show_size(concat_2)

        deconv_3 = self.deconv_3(concat_2) # -> [1, 128, 8, 16, 16]
        # utils.show_size(deconv_3)
        concat_3 = torch.cat([conv_3, deconv_3], dim=1) # -> [1, 256, 8, 16, 16]
        # utils.show_size(concat_3)

        deconv_4 = self.deconv_4(concat_3) # -> [1, 64, 16, 32, 32]
        # utils.show_size(deconv_4)

        deconv_5 = self.deconv_5(deconv_4) # -> [1, 32, 32, 64, 64]
        # utils.show_size(deconv_5)

        deconv_6 = torch.tanh(self.deconv_6(deconv_5)) # -> [1, 3, 32, 128, 128]
        # utils.show_size(deconv_6)

        return deconv_6


class RefineNetDiscriminator(nn.Module):
    """
        RefineNetDiscriminator
    """
    def __init__(self):
        super(RefineNetDiscriminator, self).__init__()
        self.in_dim = 3
        self.out_dim = 1
        self.filters = 32
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.conv_1 = utils.conv_first(self.in_dim, self.filters) # 32
        self.conv_2 = utils.conv_421(self.filters, self.filters*2, self.activation) # 64
        self.conv_3 = utils.conv_421(self.filters*2, self.filters*4, self.activation) # 128
        self.conv_4 = utils.conv_421(self.filters*4, self.filters*8, self.activation) # 256
        self.conv_5 = utils.conv_421(self.filters*8, self.filters*16, self.activation) # 512
        self.conv_6 = nn.Conv3d(self.filters*16, self.out_dim, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        # print("D: Down sampling")
        conv_1 = self.conv_1(x) # -> [1, 32, 32, 64, 64]
        # utils.show_size(conv_1)
        conv_2 = self.conv_2(conv_1) # -> [1, 64, 16, 32, 32]
        # utils.show_size(conv_2)
        conv_3 = self.conv_3(conv_2) # -> [1, 128, 8, 16, 16]
        # utils.show_size(conv_3)
        conv_4 = self.conv_4(conv_3) # -> [1, 256, 4, 8, 8]
        # utils.show_size(conv_4)
        conv_5 = self.conv_5(conv_4) # -> [1, 512, 2, 4, 4]
        # utils.show_size(conv_5)
        conv_6 = torch.sigmoid(self.conv_6(conv_5)) # -> [1, 1, 2, 4, 4]
        # utils.show_size(conv_6)

        return [conv_1, conv_3, conv_6.view(-1, 1).squeeze(1)]


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 128
    T = 32 # times of duplicated
    x = torch.Tensor(1, 3, T, image_size, image_size)
    x.to(device)
    print("x size: {}".format(x.size()))
    
    refineNetG = RefineNetGenerator(in_dim=3, out_dim=3, filters=32)
    refineNetD = RefineNetDiscriminator()
    
    out = refineNetG(x)
    # print("BaseNetGenerator output size: {}".format(out.size()))
    out = refineNetD(x)
    # print("BaseNetDiscriminartor output size:\n {},\n {},\n {}".format(down_1.size(), down_3.size(), out.size()))