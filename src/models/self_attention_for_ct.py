import torch
import torch.nn as nn
try:
    from . import utils
except:
    import utils
from sys import exit


class SelfAttentionForHW(nn.Module):
    def __init__(self, in_dim, activation):
        super(SelfAttentionForHW, self).__init__()
        self.in_dim = in_dim
        self.activation = activation

        self.query_conv = nn.Conv3d(in_channels=self.in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=self.in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=self.in_dim, out_channels=in_dim//8, kernel_size=1)

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
        print("Query")
        proj_query = self.query_conv(x) # -> [1, 16, 32, 32, 32]
        print(proj_query.size())
        bs, channel, t, width, height = proj_query.size()
        proj_query = proj_query.view(bs, channel, t, -1) # -> [1, C, T, W*H]
        print(proj_query.size())
        proj_query = proj_query.view(bs, -1, proj_query.size()[-1]) # -> [B, C*T, H*W] //
        print(proj_query.size())

        print("Key")
        proj_key = self.key_conv(x)
        print(proj_key.size())
        proj_key = proj_key.view(bs, channel, t, -1) # -> [1, C, T, W*H]
        print(proj_key.size())
        proj_key = proj_key.view(bs, -1, proj_key.size()[-1]).permute(0,2,1) # -> [B, H*W, C*T] //
        print(proj_key.size())
        print("attention")
        energy = torch.bmm(proj_query, proj_key) # -> [B, C*T, C*T]
        print(energy.size())
        attention = self.softmax(energy) # -> [B, C*T, C*T]は空間でのattention? ## TODO: [B, C*T, C*T]にして，時間軸でattentionを考える.　そもそもこの前処理でいいのか？
        # 滝とか川は流れる場所によって速度が違う.
        print(attention.size())

        print("Value")
        proj_value = self.value_conv(x)
        print(proj_value.size())
        bs, channel, t, width, height = proj_value.size()
        proj_value = proj_value.view(bs, channel, t, -1).view(bs, -1, width*height).permute(0,2,1) # -> [B, C*T, W*H]
        print(proj_value.size())


        out = torch.bmm(proj_value, attention.permute(0,2,1))
        
        bs, channel, t, width, height = x.size()
        print(out.permute(0,2,1).size())
        out = out.view(bs, channel, t, width, height) # -> [B, C, T, W, H]
        # out = self.gamma * out + x
        return attention, "a"
        
if __name__ == "__main__":
    # First Attn Block is [1, 128, 32, 32, 32]
    x = torch.Tensor(1, 128, 32, 32, 32)
    self_attn = SelfAttentionForHW(128, 'relu')
    output, attention = self_attn(x)
    # print(output.size()) # -> [1, 128, 32, 32, 32]
    