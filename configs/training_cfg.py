device = "cuda"

# Content Encoder参数
ce_args = {
    'input_dim': 256,
    'embed_dim': 128,
    'residual_size': 196,
    'skip_size': 196,
    'blocks': 3,
    'dilation_depth': 6
}

# Speaker Encoder参数
se_args = {
    'convdim_change': [256, 264, 296, 328, 360, 392],
    'convsize_change': [5, 5, 5, 3, 3],
    'embed_dim': 196,
    'qk_dim': 192,
    'head': 4,
    'dim_feedforward': 224
}

# 解码器参数
d_args = {}

# mel config
mel_cfg = {
    'sample_rate': 22050,
    'n_fft':2048,              #  creates n_fft // 2 + 1 bins.
    'win_length':1024,         #  default = n_fft
    'hop_length':512,         #  default = win_length // 2
    'n_mels':256,
    'normalized':False,
    'norm':'slaney',
    'mel_scale':'slaney'      # librosa mel_to_audio
}



# sample_rate=22050
# n_fft=2048              #  creates n_fft // 2 + 1 bins.
# win_length=1024         #  default = n_fft
# hop_length=512          #  default = win_length // 2
# n_mels=256
# normalized=False
# norm='slaney'
# mel_scale='slaney'      # librosa mel_to_audio的默认