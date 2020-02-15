import torch
import torch.nn as nn
try:
    from . import utils
except:
    import utils
from sys import exit


class SelfAttentionForHW(nn.Module):
    def __init__(self, in_dim, activation, c_cal=8):
        super(SelfAttentionForHW, self).__init__()
        self.in_dim = in_dim
        self.activation = activation
        
        c_cal = c_cal
        self.query_conv = nn.Conv3d(in_channels=self.in_dim, out_channels=in_dim//c_cal, kernel_size=1)
        # self.query_conv_2d = nn.Conv2d(in_channels=self.in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=self.in_dim, out_channels=in_dim//c_cal, kernel_size=1)
        # self.key_conv_2d = nn.Conv2d(in_channels=self.in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=self.in_dim, out_channels=in_dim, kernel_size=1)
        # self.value_conv_2d = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1) # TODO: なぜdim=-1なの？

    def forward(self, x):
        """
            inputs:
                x : input feature maps([B, C, T, W, H])
            returns:
                out : self attention value + input feature
                attention [B x T x N x N] (N is W * H)
        """
        bs, channel, t, width, height = x.size()
        # Query Convolution.
        proj_query = self.query_conv(x) # -> [1, 16, 32, 32, 32]
        # print('qurey conv: ', proj_query.size())
        bs, channel, t, width, height = proj_query.size()
        proj_query = proj_query.view(bs, channel, t, -1) # -> [1, C, T, W*H]
        # print(proj_query.size())
        proj_query = proj_query.view(bs, -1, proj_query.size()[-1]).permute(0,2,1) # -> [B, H*W, C*T]
        # print('query: ', proj_query.size())

        proj_key = self.key_conv(x)
        # print('key conv: ', proj_key.size())
        proj_key = proj_key.view(bs, channel, t, -1) # -> [1, C, T, W*H]
        # print(proj_key.size())
        proj_key = proj_key.view(bs, -1, proj_key.size()[-1]) # -> [B, C*T, H*W]
        # print('key: ', proj_key.size())

        energy = torch.bmm(proj_query, proj_key) # -> [B, H*W, H*W]
        # print(energy.size())
        attention = self.softmax(energy) # -> [B, H*W, H*W]は空間でのattention? ## TODO: [B, C*T, C*T]にして，時間軸でattentionを考える.　そもそもこの前処理でいいのか？
        # 滝とか川は流れる場所によって速度が違う.
        # print('attention: ', attention.size())

        proj_value = self.value_conv(x)
        # proj_value = self.sigmoid(proj_value)
        # print('value conv: ', proj_value.size())
        bs, channel, t, width, height = proj_value.size()
        proj_value = proj_value.view(bs, channel, t, -1).view(bs, -1, width*height) # -> [B, C*T, W*H]
        # print('value: ', proj_value.size())

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        # print('out bmm: ', out.size())
        
        bs, channel, t, width, height = x.size()
        out = out.view(bs, channel, t, width, height) # -> [B, C, T, W, H]
        # print('out view: ', out.size())

        out = self.gamma * out + x
        # print(out.size())
        return out, attention
        
if __name__ == "__main__":
    # First Attn Block is [1, 128, 32, 32, 32]
    x = torch.Tensor(1, 128, 32, 32, 32)
    self_attn = SelfAttentionForHW(128, 'relu')
    output, attention = self_attn(x)
    # print(output.size()) # -> [1, 128, 32, 32, 32]
    # print(attention.size())