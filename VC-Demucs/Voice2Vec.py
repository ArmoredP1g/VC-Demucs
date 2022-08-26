"""
Embedding sound samples into vectors
The input is the STFT spectrum
"""
import torch
import torch.nn as nn


class Voice2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        # STFT
        # LSTM layer * 3
        # attentive pooling