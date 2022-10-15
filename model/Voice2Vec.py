"""
Embedding sound samples into vectors
The input is the Mel spectrum
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from model.WaveNet import WaveNet
from configs.training_cfg import device

class oneside_pad(nn.Module):
    def __init__(self, padding_size, dim, left_padding=True):
        super().__init__()
        self.padding_size = padding_size
        self.left_padding = left_padding
        self.dim = dim
    
    def forward(self,x):
        shape = list(x.shape)
        shape[self.dim] = self.padding_size
        if self.left_padding:
            return torch.cat((torch.zeros(shape).to(device),x), self.dim)
        else:
            return torch.cat(x,(torch.zeros(shape).to(device)), self.dim)


class Voice2Vec(nn.Module):
    '''
    @todo 尝试用lstm代替 naive_attn测试效果
    '''
    def __init__(self, attn=True):
        super().__init__()
        self.attn_flag = attn
        self.soft = nn.Softmax(dim=2)
        self.WaveNet = WaveNet(input_size=80,
                                out_size=80,
                                residual_size=64,
                                skip_size=60,
                                blocks=2,
                                dilation_depth=5)

        self.conv = nn.Sequential(
                            nn.AvgPool1d(2,2),
                            nn.Conv1d(80,100,5,1,2),
                            nn.ELU(),
                            nn.AvgPool1d(2,2),
                            nn.Conv1d(100,128,5,1,2)
                        )

        self.naive_attn = nn.Sequential(
                            nn.Linear(128, 64, bias=False),
                            nn.ELU(),
                            nn.Linear(64, 1, bias=False)
                        )

    def weight_init(self):
        pass
    
    def forward(self, mel):
        '''
        input: spk*uttr, mel_num, lenth
        return: spk*uttr * 128
        '''
        emb = self.WaveNet(mel)
        emb = self.conv(emb)
        b, d, l = emb.shape
        emb_ = emb.permute(0,2,1).reshape(b*l, d)
        weight = self.naive_attn(emb_).view(b, l, 1).permute(0,2,1)
        weight = self.soft(weight)
        emb = emb * weight
        return F.normalize(emb.sum(2), p=2, dim=1), weight # normalize vec


class Voice2Vec_lstm(nn.Module):
    '''
    @todo 尝试用lstm代替 naive_attn测试效果
    '''
    def __init__(self, step=160, attn=True):
        super().__init__()
        self.step = step
        self.attn_flag = attn
        self.soft = nn.Softmax(dim=2)

        self.naive_attn = nn.Sequential(
                    nn.Linear(128, 64, bias=False),
                    nn.ELU(),
                    nn.Linear(64, 1, bias=False),
                    nn.ELU(),
                )

        self.lstm = nn.LSTM(input_size=80, hidden_size=128, batch_first=True, num_layers=3)

    def weight_init(self):
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)
    
    def forward(self, mel):
        '''
        input: spk*uttr, mel_num, lenth
        return: spk*uttr * 512
        '''
        _, _, l = mel.shape
        mel = mel.permute(0,2,1)
        emb = torch.Tensor().to(device)
        i = 0
        while i + self.step // 2 <= l:
            o,(h,c) = self.lstm(mel[:,i:i+self.step])
            emb = torch.cat((emb, h[-1].unsqueeze(1)), 1)
            i += self.step // 2

        # emb: batch, len, mel

        if self.attn_flag:
            b, l, m = emb.shape
            weight = self.naive_attn(emb.view(b*l, m)).view(b,l,1)
            weight = self.soft(weight)
            emb = emb * weight
            return F.normalize(emb.sum(dim=1), p=2, dim=1), weight # normalize vec

        return F.normalize(emb.mean(dim=1), p=2, dim=1), None # normalize vec

def loss_trunc(x):
    '''
    When calculating Cross loss, if the similarity is less than 0
    the optimization will not apply.
    '''
    m = x > 0
    return (0.31666+0.51-torch.log(-(0.5*x-0.51)))*m
    # return 0.31666+0.51-torch.log(-(0.5*x-0.51))

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
        cross_loss += loss_trunc(dot.max(dim=0)[0]).sum()

    return center_loss, cross_loss