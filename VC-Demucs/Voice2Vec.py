"""
Embedding sound samples into vectors
The input is the Mel spectrum
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

#   残差卷积结构
class _Res1D_convsizefixed_v1(nn.Module):
    def __init__(self,in_channels, out_channels, convsize):
        super().__init__()
        self.overlapTile = nn.ReflectionPad1d(int(convsize/2))  #对称填充
        # bn layers
        self.BN_1 = nn.BatchNorm1d(in_channels)
        self.BN_2 = nn.BatchNorm1d(out_channels)

        self.conv_1 = nn.Conv1d(in_channels,in_channels,convsize,1)
        self.conv_2 = nn.Conv1d(in_channels,out_channels,convsize,1)

        # 1*1 conv
        if in_channels != out_channels:
            self.bottleneck_conv = nn.Conv1d(in_channels,out_channels,1,1)
        else:
            self.bottleneck_conv = None

    
    def forward(self,x):
        X = x
        output = self.overlapTile(x)
        output = self.conv_1(output)
        output = self.BN_1(output)
        output = torch.relu(output)
        output = self.overlapTile(output)
        output = self.conv_2(output)
        output = self.BN_2(output)

        if self.bottleneck_conv:
            X = self.bottleneck_conv(x)
        
        return output + X

from turtle import forward


class Voice2Vec(nn.Module):
    def __init__(self, L2=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=2)
        self.RConv = nn.Sequential(
                        _Res1D_convsizefixed_v1(80, 128, 5),
                        nn.LeakyReLU(0.1),
                        _Res1D_convsizefixed_v1(128,256, 5),
                        nn.LeakyReLU(0.1),
                        _Res1D_convsizefixed_v1(256,512, 5),
                        nn.LeakyReLU(0.05),
                        )

        self.naive_attn = nn.Sequential(
                            nn.Linear(512, 128),
                            nn.LeakyReLU(0.1),
                            nn.Linear(128, 128),
                            nn.LeakyReLU(0.1),
                            nn.Linear(128, 32),
                            nn.LeakyReLU(0.1),
                            nn.Linear(32, 1),
                        )

    def weight_init(self):
        pass
    
    def forward(self, mel):
        emb = self.RConv(mel)
        b, d, l = emb.shape
        emb_ = emb.permute(0,2,1).view(b*l, d)
        weight = self.naive_attn(emb_).view(b, l, 1).permute(0,2,1)
        weight = self.softmax(weight)
        emb = emb * weight
        return F.normalize(emb.sum(2), p=2, dim=1) # normalize vec
