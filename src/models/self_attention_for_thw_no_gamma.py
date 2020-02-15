import torch
import torch.nn as nn
try:
    from . import utils
except:
    import utils
from sys import exit


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_dim, activation, c_cal=8):
        super(SelfAttentionBlock, self).__init__()
        self.in_dim = in_dim
        self.activation = activation
        
        c_cal = c_cal
        self.query_conv = nn.Conv3d(in_channels=self.in_dim, out_channels=in_dim//c_cal, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=self.in_dim, out_channels=in_dim//c_cal, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=self.in_dim, out_channels=in_dim, kernel_size=1)

        # self.gamma = nn.Parameter(torch.zeros(1))
        self.W = nn.Sequential(
                nn.Conv3d(in_channels=self.in_dim, out_channels=self.in_dim,
                        kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(self.in_dim)
            )
        nn.init.constant_(self.W[0].weight, 0)
        nn.init.constant_(self.W[0].bias, 0)
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
        proj_query = proj_query.view(bs, channel, -1) # -> [1, C, T*W*H]
        # print(proj_query.size())
        proj_query = proj_query.permute(0,2,1) # -> [B, T*H*W, C]

        proj_key = self.key_conv(x)
        # print('key conv: ', proj_key.size())
        proj_key = proj_key.view(bs, channel, t, -1) # -> [1, C, T, W*H]
        # print(proj_key.size())
        proj_key = proj_key.view(bs, channel, -1) # -> [1, C, T*W*H]
        # print(proj_key.size())
        # proj_key = proj_key.view(bs, -1, proj_key.size()[-1]) # -> [B, C*T, H*W]
        # print('key: ', proj_key.size())

        energy = torch.bmm(proj_query, proj_key) # -> [B, T*H*W, T*H*W]
        # print(energy.size())
        attention = self.softmax(energy) # -> [B, T*H*W, T*H*W]
        # 滝とか川は流れる場所によって速度が違う.
        # print('attention: ', attention.size())

        proj_value = self.value_conv(x)
        # proj_value = self.sigmoid(proj_value)
        # print('value conv: ', proj_value.size())
        bs, channel, t, width, height = proj_value.size()
        proj_value = proj_value.view(bs, channel, t, -1).view(bs, channel, -1) # -> [B, C, T*W*H]

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        # print('out bmm: ', out.size())
        
        bs, channel, t, width, height = x.size()
        out = out.view(bs, channel, t, width, height) # -> [B, C, T, W, H]

        out = self.W(out)
        out = out + x

        # out = self.gamma * out + x
        # print(out.size())
        return out, attention
        
if __name__ == "__main__":
    # First Attn Block is [1, 128, 32, 32, 32]
    x = torch.Tensor(1, 128, 32, 32, 32)
    print(f"Input: {x.size()}")
    self_attn = SelfAttentionBlock(128, 'relu', c_cal=1)
    output, attention = self_attn(x)
    # print(output.size()) # -> [1, 128, 32, 32, 32]
    # print(attention.size())