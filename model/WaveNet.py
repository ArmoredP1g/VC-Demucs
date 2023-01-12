"""
https://zhuanlan.zhihu.com/p/231108835
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.training_cfg import device

class CausalConv1d(nn.Module):
    """
    for WaveNet
    Input and output sizes will be the same.
    """
    def __init__(self, ch_in, ch_out, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(ch_in, ch_out, kernel_size, padding=self.pad, dilation=dilation)

    def forward(self, x):
        x = self.conv1(x)
        x = x[..., :-self.pad]  # drop the right side padding
        return x

class ResidualLayer(nn.Module): 
    def __init__(self, residual_size, skip_size, dilation):
        """
        for WaveNet
        args:
            residual_size:残差通道
            skip_size:跳跃通道
            dilation:空洞尺寸
        """
        super(ResidualLayer, self).__init__()
        self.conv_filter = CausalConv1d(residual_size, residual_size,
                                         kernel_size=2, dilation=dilation)
        self.conv_gate = CausalConv1d(residual_size, residual_size,
                                         kernel_size=2, dilation=dilation)        
        self.resconv1_1 = nn.Conv1d(residual_size, residual_size, kernel_size=1)
        self.skipconv1_1 = nn.Conv1d(residual_size, skip_size, kernel_size=1)
        
    def forward(self, x):
        conv_filter = self.conv_filter(x)
        conv_gate = self.conv_gate(x)  
        fx = torch.tanh(conv_filter) * torch.sigmoid(conv_gate)
        fx = self.resconv1_1(fx) 
        skip = self.skipconv1_1(fx) 
        residual = fx + x  
        #residual=[batch,residual_size,seq_len]  skip=[batch,skip_size,seq_len]
        return skip, residual

class DilatedStack(nn.Module):
    def __init__(self, residual_size, skip_size, dilation_depth):
        super(DilatedStack, self).__init__()
        residual_stack = [ResidualLayer(residual_size, skip_size, 2**layer)
                         for layer in range(dilation_depth)]
        self.residual_stack = nn.ModuleList(residual_stack)
        
    def forward(self, x):
        skips = []
        for layer in self.residual_stack:
            skip, x = layer(x)
            skips.append(skip.unsqueeze(0))
            #skip =[1,batch,skip_size,seq_len]
        return torch.cat(skips, dim=0), x  # [layers,batch,skip_size,seq_len]

class WaveNet(nn.Module):
    def __init__(self,input_size,out_size, residual_size, skip_size, blocks, dilation_depth):
        '''
        args:
            @input_size: input_channels,
            @out_size: output_channels,
            @residual_size: hidden_size for residual,
            @skip_size: ..,
            @dilation_depth: 
        '''
        super(WaveNet, self).__init__()
        self.input_conv = CausalConv1d(input_size,residual_size, kernel_size=2)        
        self.dilated_stacks = nn.ModuleList(
            [DilatedStack(residual_size, skip_size, dilation_depth)
             for cycle in range(blocks)]
        )

        self.convout_1 = nn.Conv1d(skip_size, out_size, kernel_size=1)
        self.convout_2 = nn.Conv1d(out_size, out_size, kernel_size=1)

    def forward(self, x):
        x = self.input_conv(x) # [batch,residual_size, seq_len]             
        skip_connections = []
        for cycle in self.dilated_stacks:
            skips, x = cycle(x)             
            skip_connections.append(skips)

        ## skip_connection=[total_layers,batch,skip_size,seq_len]
        skip_connections = torch.cat(skip_connections, dim=0)        

        # gather all output skip connections to generate output, discard last residual output
        out = skip_connections.sum(dim=0) # [batch,skip_size,seq_len]
        out = F.relu(out)
        out = self.convout_1(out) # [batch,out_size,seq_len]
        out = F.relu(out)

        out=self.convout_2(out)
        return out     


class Conditional_WaveNet(nn.Module):
    def __init__(self,input_size,out_size, residual_size, skip_size, blocks, dilation_depth):
        '''
        apply ADAIN on every skip output
        args:
            @input_size: input_channels,
            @out_size: output_channels,
            @residual_size: hidden_size for residual,
            @skip_size: ..,
            @dilation_depth: 
        '''
        super(Conditional_WaveNet, self).__init__()
        self.input_conv = CausalConv1d(input_size,residual_size, kernel_size=2)        
        self.dilated_stacks = nn.ModuleList(
            [DilatedStack(residual_size, skip_size, dilation_depth)
             for cycle in range(blocks)]
        )

        self.convout_1 = nn.Conv1d(skip_size, out_size, kernel_size=1)
        self.convout_2 = nn.Conv1d(out_size, out_size, kernel_size=1)

        self.IN_input = nn.InstanceNorm1d(input_size, track_running_stats=True)
        self.IN_res = nn.ModuleList(
            [nn.InstanceNorm1d(residual_size, track_running_stats=True)
             for cycle in range(blocks)]
        )

    def forward(self, x, condition_gamma, condition_beta):
        x = self.IN_input(x)
        x = self.input_conv(x) # [batch,residual_size, seq_len]             
        skip_connections = []
        for cycle in range(self.dilated_stacks.__len__()):
            skips, x = self.dilated_stacks[cycle](x)            
            x = self.IN_res[cycle](x)              # remove speaker feature
            if condition_beta != None and condition_gamma != None:
                x = (x.permute(2,0,1)*condition_gamma + condition_beta).permute(1,2,0)  # introduce speaker condition
                # else do in only
            skip_connections.append(skips)

        ## skip_connection=[total_layers,batch,skip_size,seq_len]
        skip_connections = torch.cat(skip_connections, dim=0)        

        # gather all output skip connections to generate output, discard last residual output
        out = skip_connections.sum(dim=0) # [batch,skip_size,seq_len]
        out = F.relu(out)
        out = self.convout_1(out) # [batch,out_size,seq_len]
        out = F.relu(out)

        out=self.convout_2(out)
        return out   