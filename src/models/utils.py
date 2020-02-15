import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

"""
    New functions
"""

def conv_first(in_dim, out_dim, activation, use_spectral_norm):
    if use_spectral_norm:
        return nn.Sequential(
            spectral_norm(nn.Conv3d(in_dim, out_dim, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1))),
            # spectral_norm(nn.Conv3d(in_dim, out_dim, kernel_size=7, stride=1, padding=3)),
            nn.BatchNorm3d(out_dim),
            # activation,
        )
    else:
        return nn.Sequential(
            spectral_norm(nn.Conv3d(in_dim, out_dim, kernel_size=(3,4,4), stride=(1,2,2), padding=1)),
            # nn.Conv3d(in_dim, out_dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm3d(out_dim),
            # activation,
        )

def conv_6(in_dim, out_dim, activation, use_spectral_norm):
    if use_spectral_norm:
        return nn.Sequential(
            spectral_norm(nn.Conv3d(in_dim, out_dim, kernel_size=(2,4,4), stride=(1,2,2), padding=(0,1,1))),
            # nn.BatchNorm3d(out_dim),
            activation,
        )
    else:
        return nn.Sequential(
            spectral_norm(nn.Conv3d(in_dim, out_dim, kernel_size=(3,4,4), stride=(1,2,2), padding=1)),
            # nn.BatchNorm3d(out_dim),
            activation,
        )


# def conv_first_encdec_InstanceNorm(in_dim, out_dim, activation, use_spectral_norm):
#     if use_spectral_norm:
#         print("----------------------------------------------------------------------")
#         return nn.Sequential(
#             # spectral_norm(nn.Conv3d(in_dim, out_dim, kernel_size=(3,4,4), stride=(1,2,2), padding=1))
#             spectral_norm(nn.Conv3d(in_dim, out_dim, kernel_size=7, stride=1, padding=3)),
#             # nn.BatchNorm3d(out_dim),
#             nn.InstanceNorm3d(out_dim),
#             activation,
#         )
#     else:
#         return nn.Sequential(
#             # spectral_norm(nn.Conv3d(in_dim, out_dim, kernel_size=(3,4,4), stride=(1,2,2), padding=1))
#             nn.Conv3d(in_dim, out_dim, kernel_size=7, stride=1, padding=3),
#             nn.BatchNorm3d(out_dim),
#             activation,
#         )


# def conv_first(in_dim, out_dim, use_spectral_norm):
#     if use_spectral_norm:
#         print("----------------------------------------------------------------------")
#         return nn.Sequential(
#             spectral_norm(nn.Conv3d(in_dim, out_dim, kernel_size=(3,4,4), stride=(1,2,2), padding=1))
#         )
#     else:
#         return nn.Sequential(
#             nn.Conv3d(in_dim, out_dim, kernel_size=(3,4,4), stride=(1,2,2), padding=1)
#         )


def conv_421(in_dim, out_dim, activation, use_spectral_norm):
    if use_spectral_norm:
        return nn.Sequential(
            spectral_norm(nn.Conv3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm3d(out_dim),
            activation,
        )
    else:
        return nn.Sequential(
            (nn.Conv3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm3d(out_dim),
            activation,
        )

def conv_421_D(in_dim, out_dim, activation, use_spectral_norm):
    if use_spectral_norm:
        return nn.Sequential(
            spectral_norm(nn.Conv3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)),
            activation,
        )
    else:
        return nn.Sequential(
            (nn.Conv3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)),
            activation,
        )

def conv_421_InstanceNorm(in_dim, out_dim, activation, use_spectral_norm):
    if use_spectral_norm:
        return nn.Sequential(
            spectral_norm(nn.Conv3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm3d(out_dim),
            nn.InstanceNorm3d(out_dim),
            activation,
        )
    else:
        return nn.Sequential(
            (nn.Conv3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm3d(out_dim),
            nn.InstanceNorm3d(out_dim),
            activation,
        )

def conv_last(in_dim, out_dim, use_spectral_norm):
    if use_spectral_norm:
        return nn.Sequential(
            spectral_norm(nn.Conv3d(in_dim, out_dim, kernel_size=(2,4,4), stride=(1,2,2), padding=(0,1,1)))
        )
    else:
        return nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=(2,4,4), stride=1, padding=0)
        )
        

def deconv_first(in_dim, out_dim, use_spectral_norm):
    if use_spectral_norm:
        return nn.Sequential(
            spectral_norm(nn.ConvTranspose3d(in_dim, out_dim, kernel_size=(4,4,4), stride=(1,2,2), padding=1)),
            nn.BatchNorm3d(out_dim)
        )
    else:
        return nn.Sequential(
            spectral_norm(nn.ConvTranspose3d(in_dim, out_dim, kernel_size=(4,4,4), stride=1, padding=0)),
            nn.BatchNorm3d(out_dim)
        )
    

def deconv_421(in_dim, out_dim, activation, use_spectral_norm):
    if use_spectral_norm:
        return nn.Sequential(
            spectral_norm(nn.ConvTranspose3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm3d(out_dim),
            activation
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(out_dim),
            activation
        )

def deconv_421_InstanceNorm(in_dim, out_dim, activation, use_spectral_norm):
    if use_spectral_norm:
        return nn.Sequential(
            spectral_norm(nn.ConvTranspose3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm3d(out_dim),
            nn.InstanceNorm3d(out_dim),
            activation
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm3d(out_dim),
            nn.InstanceNorm3d(out_dim),
            activation
        )

def deconv_last(in_dim, out_dim, use_spectral_norm):
    if use_spectral_norm:
        return nn.Sequential(
            spectral_norm(nn.ConvTranspose3d(in_dim, out_dim, kernel_size=(3,4,4), stride=(1,2,2), padding=1))
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=(3,4,4), stride=(1,2,2), padding=1)
        )




# """---- deprecated ----"""
# def max_pooling_3d():
#     return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


# def conv_block_2_3d(in_dim, out_dim, activation):
#     return nn.Sequential(
#         conv_block_3d(in_dim, out_dim, activation),
#         conv_block_3d(out_dim, out_dim, activation)
#     )

# def conv_trans_block_3d(in_dim, out_dim, activation):
#     return nn.Sequential(
#         nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
#         nn.BatchNorm3d(out_dim),
#         activation
#     )


# def conv_block_2_3d_without_BN_and_Act(in_dim, out_dim, activation):
#     return nn.Sequential(
#         nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
#         conv_block_3d(out_dim, out_dim, activation)
#     )


def show_size(x):
    print(x.size())

# """
# For Generator.
# """
# def trans_conv_block(in_dim, out_dim, activation):
#     return nn.Sequential(
#         trans_conv_421(in_dim, out_dim, activation),
#         trans_conv_311(out_dim, out_dim, activation)
#         # trans_conv_311(in_dim, out_dim, activation)
#     )

# def trans_conv_block_first(in_dim, out_dim, activation):
#     return nn.Sequential(
#         trans_conv_311(in_dim, out_dim, activation),
#         trans_conv_311(out_dim, out_dim, activation)
#     )

# def conv_trans_block(in_dim, out_dim, activation):
#     return nn.Sequential(
#         trans_conv_421(in_dim, out_dim, activation),
#         trans_conv_311(out_dim, out_dim, activation)
#     )


# def trans_conv_421(in_dim, out_dim, activation):
#     return nn.Sequential(
#         nn.ConvTranspose3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
#         nn.BatchNorm3d(out_dim),
#         activation
#     )

# def trans_conv_311(in_dim, out_dim, activation):
#     return nn.Sequential(
#         nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm3d(out_dim),
#         activation
#     )


# def conv_block_bridge(in_dim, out_dim, activation):
#     return nn.Sequential(
#         conv_311(in_dim, out_dim, activation),
#         conv_311(out_dim, out_dim, activation)
#     )

# """
# For Discriminator.
# """
# def conv_block(in_dim, out_dim, activation):
#     return nn.Sequential(
#         conv_421(in_dim, out_dim, activation),
#         conv_311(out_dim, out_dim, activation)
#     )


# def conv_421(in_dim, out_dim, activation):
#     return nn.Sequential(
#         nn.Conv3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
#         nn.BatchNorm3d(out_dim),
#         activation,
#     )

# def conv_311(in_dim, out_dim, activation):
#     return nn.Sequential(
#         nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm3d(out_dim),
#         activation
#     )


# def conv_block_without_BN_and_Acti(in_dim, out_dim, activation):
#     return nn.Sequential(
#         nn.Conv3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
#         nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm3d(out_dim),
#         activation
#     )


# def conv_block_without_BN_and_Acti1(in_dim, hidden_dim, out_dim, activation):
#     return nn.Sequential(
#         nn.Conv3d(in_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
#         nn.Conv3d(hidden_dim, out_dim, kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm3d(out_dim),
#         activation
#     )


# def conv_block_3d(in_dim, out_dim, activation):
#     return nn.Sequential(
#         nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm3d(out_dim),
#         activation
#     )

if __name__ == '__main__':
    conv = conv_421(32, 64, nn.ReLU(), True)
    conv = conv_421(32, 64, nn.ReLU(), False)