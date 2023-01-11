import torch
import torch.nn as nn
import torch.nn.functional as F
from model.WaveNet import Conditional_WaveNet
from model.ProbSparse_Self_Attention import Self_Attention_Decoder

def Reparameterize(mu, logvar):
    # 从 Normal(μ, σ^2)中采样一个Z
    # 相当于从Normal(0, 1)采样一个ε
    # Z = μ + ε × σ
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

class Res1D_convsizefixed(nn.Module):
    def __init__(self,in_channels, out_channels, convsize):
        super().__init__()
        self.overlapTile = nn.ReflectionPad1d(convsize//2)  #对称填充

        self.conv_1 = nn.Conv1d(in_channels,in_channels,convsize,1,bias=False)
        self.conv_2 = nn.Conv1d(in_channels,out_channels,convsize,1,bias=False)

        # 1*1 conv
        if in_channels != out_channels:
            self.bottleneck_conv = nn.Conv1d(in_channels,out_channels,1,1)
        else:
            self.bottleneck_conv = None

    
    def forward(self,x):
        X = x
        output = self.overlapTile(x)
        output = self.conv_1(output)
        output = torch.relu(output)
        output = self.overlapTile(output)
        output = self.conv_2(output)

        if self.bottleneck_conv:
            X = self.bottleneck_conv(x)
        
        return output + X

class Content_Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # default_args
        self.args = {
            'input_dim': 80,
            'embed_dim': 128,
            'residual_size': 196,
            'skip_size': 196,
            'blocks': 3,
            'dilation_depth': 7
        }

        self.args.update(kwargs)

        self.wave_net = Conditional_WaveNet(self.args['input_dim'],
                                self.args['embed_dim']*2,       # for μ and σ
                                self.args['residual_size'],
                                self.args['skip_size'],
                                self.args['blocks'],
                                self.args['dilation_depth'])
        self.conv_miu = nn.Conv1d(self.args['embed_dim']*2, self.args['embed_dim'], 1, 1)
        self.conv_logvar = nn.Conv1d(self.args['embed_dim']*2, self.args['embed_dim'], 1, 1)
        self.IN = nn.InstanceNorm1d(self.args['embed_dim']*2, track_running_stats=True)

    def forward(self, x):
        x = torch.relu(self.wave_net(x, None, None))    # batch, embed, len
        
        x = self.IN(x)

        miu = self.conv_miu(x)
        logvar = self.conv_logvar(x)

        return miu, logvar


class Speaker_Encoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        # default_args
        self.args = {
            'convdim_change': [80, 128, 160, 192, 224, 256],
            'convsize_change': [5, 5, 5, 3, 3],
            'embed_dim': 128,
            'qk_dim': 192,
            'head': 4,
            'dim_feedforward': 224
        }

        self.args.update(kwargs)

        self.avg_pool = nn.AvgPool1d(2,2)
        self.resconv = []
        for i in range(self.args['convsize_change'].__len__()):
            self.resconv.append(Res1D_convsizefixed(self.args['convdim_change'][i],self.args['convdim_change'][i+1],self.args['convsize_change'][i]))
        self.resconv = nn.ModuleList(self.resconv)

        self.attn_layer = Self_Attention_Decoder(
            input_dim=self.args['convdim_change'][-1],
            output_dim=self.args['embed_dim']*2,
            qk_dim=self.args['qk_dim'],
            heads=self.args['head'],
            dim_feedforward=self.args['dim_feedforward']
        )

    def forward(self, x):
        for c in self.resconv:
            x = self.avg_pool(torch.relu(c(x))) # b, dim, len
        
        x = self.attn_layer(x.transpose(1,2))
        gamma = x[:,0:self.args['embed_dim']]
        beta = x[:,self.args['embed_dim']:]

        return gamma, beta
        


    
# class Decoder(nn.Module):

    