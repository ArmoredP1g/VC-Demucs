device = "cuda"

# Content Encoder参数
ce_args = {
    'input_dim': 80,
    'embed_dim': 128,
    'residual_size': 196,
    'skip_size': 196,
    'blocks': 2,
    'dilation_depth': 7
}

# Speaker Encoder参数
se_args = {
    'convdim_change': [80, 128, 160, 192, 224, 256],
    'convsize_change': [5, 5, 5, 3, 3],
    'embed_dim': 128,
    'qk_dim': 192,
    'head': 4,
    'dim_feedforward': 224
}

# 解码器参数
d_args = {}