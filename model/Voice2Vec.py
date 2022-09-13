"""
Embedding sound samples into vectors
The input is the Mel spectrum
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from configs.training_cfg import device

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


class Voice2Vec_abandoned(nn.Module):
    '''
    @todo 尝试用lstm代替 naive_attn测试效果
    '''
    def __init__(self):
        super().__init__()

        self.soft = nn.Softmax(dim=2)
        self.RConv = nn.Sequential(
                        _Res1D_convsizefixed_v1(80, 160, 5),
                        nn.ELU(),
                        nn.AvgPool1d(2,2),
                        _Res1D_convsizefixed_v1(160,320, 5),
                        nn.ELU(),
                        nn.AvgPool1d(2,2),
                        _Res1D_convsizefixed_v1(320,400, 5),
                        nn.ELU(),
                        nn.AvgPool1d(2,2),
                        _Res1D_convsizefixed_v1(400,512, 5),
                        nn.ELU(),
                        nn.AvgPool1d(2,2),
                        )

        self.naive_attn = nn.Sequential(
                            nn.Linear(512, 128),
                            nn.ELU(),
                            nn.Linear(128, 128),
                            nn.ELU(),
                            nn.Linear(128, 1)
                        )

    def weight_init(self):
        pass
    
    def forward(self, mel):
        '''
        input: spk*uttr, mel_num, lenth
        return: spk*uttr * 512
        '''
        emb = self.RConv(mel)
        b, d, l = emb.shape
        emb_ = emb.permute(0,2,1).reshape(b*l, d)
        weight = self.naive_attn(emb_).view(b, l, 1).permute(0,2,1)
        weight = self.soft(weight)
        emb = emb * weight
        return F.normalize(emb.sum(2), p=2, dim=1) # normalize vec

    def test(self, mel):
        emb = self.RConv(mel)
        b, d, l = emb.shape
        emb_ = emb.permute(0,2,1).reshape(b*l, d)
        weight = self.naive_attn(emb_).view(b, l, 1).permute(0,2,1)
        weight = self.soft(weight)
        emb = emb * weight
        return F.normalize(emb.sum(2), p=2, dim=1), weight # normalize vec

def loss_multiplier(x):
    return 0.31666+0.51-torch.log(-(0.5*x-0.51))
    # 0.31666


def simplified_ge2e_loss(x):
    '''
    a simplified version of ge2e_loss
    x_shape: spk * uttr * 512
    '''
    spk, uttr, _ = x.shape
    # 这里应该判断下spk uttr必须相等，但没必要

    # 求每个speaker 的 center
    centers = F.normalize(x.mean(dim=1), p=2, dim=1) # this need to renormalized
    cross_loss = torch.tensor(0.).to(device)
    center_loss = torch.tensor(0.).to(device)

    for i in range(spk):
        uttr_emb = x[i] # uttr * 512
        # for the same speaker, the similarity of embedding vectors 
        # should be increased as much as possible
        # for the different speaker should be decreased
        mat_left = uttr_emb.unsqueeze(1)    # uttr * 1 * 512
        mat_right = centers.unsqueeze(0)    # 1 * uttr * 512
        dot = (mat_left * mat_right).sum(dim=2)

        # mask_r = torch.zeros(uttr,uttr).to(device)
        # mask_r[:,i] = -1
        center_loss += -1*dot[:,i].sum()
        dot [:,i] = -1
        cross_loss += loss_multiplier(dot.max(dim=0)[0]).sum()

    return center_loss, cross_loss