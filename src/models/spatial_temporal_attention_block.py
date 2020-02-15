import torch
import torch.nn as nn
try:
    from . import utils
except:
    import utils
from sys import exit


class SpatialTemporalAttentionBlock(nn.Module):
    """
    参考: https://arxiv.org/pdf/1807.05073.pdf
    3.2 Non-local Attention Block
    上記論文では，3DResBlockの後に追加してた.
    """
    def __init__(self, in_dim):
        super(SpatialTemporalAttentionBlock, self).__init__()
        self.in_dim = in_dim

        self.query_conv = nn.Conv3d(in_channels=self.in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=self.in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=self.in_dim, out_channels=in_dim//2, kernel_size=1)
        self.up_conv = nn.Conv3d(in_channels=self.in_dim//2, out_channels=in_dim, kernel_size=1)

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
        x_b, x_c, x_t, x_w, x_h = x.size()
        query = self.query_conv(x) # -> [1, 64, 8, 32, 32]([B, C, T, W, H])
        query = query.permute(0, 2, 3, 4, 1) # -> (B, T, H, W, C)
        query = query.view(query.size()[0], -1, query.size()[2], query.size()[-1])
        query = query.view(query.size()[0], -1, query.size()[-1]) # -> [B, THW, C] //

        key = self.key_conv(x) # -> [B, C, T, W, H]
        key = key.view(key.size()[0], key.size()[1], key.size()[2], -1) # -> [B, C, T, HW]
        key = key.view(key.size()[0], key.size()[1], -1) # -> [B, C, THW] //

        attention = torch.bmm(query, key) # -> [B, THW, THW]
        attention = self.softmax(attention) # -> [B, THW, THW] //

        value = self.value_conv(x) # -> [B, C, T, W, H]
        value = value.view(key.size()[0], key.size()[1], key.size()[2], -1)
        value = value.view(key.size()[0], key.size()[1], -1) # -> [B, C, TWH] //

        out = torch.bmm(value, attention) # -> [B, C//2, TWH]
        out = out.view(x_b, x_c//2, x_t, x_w, x_h) # -> [B, C//2, T, W, H]
        out = self.up_conv(out) # -> [B, C, T, W, H]

        out = self.gamma * out + x

        return out


if __name__ == "__main__":
    # First Attn Block is [1, 128, 32, 32, 32]
    x = torch.Tensor(1, 128, 8, 32, 32)
    block = SpatialTemporalAttentionBlock(128)
    output = block(x)
    print(output.size()) # -> [1, 128, 32, 32, 32]
    