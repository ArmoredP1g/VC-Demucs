import torch
import torchaudio
from torchaudio.functional import resample
import torchaudio.transforms as transforms
import torch.nn.functional as F

def process_wav(in_path, out_path, sample_rate):
    wav, sr = torchaudio.load(in_path)
    wav = resample(wav, sr, sample_rate)
    torchaudio.save(out_path, wav, sample_rate)
    return out_path, wav.size(-1) / sample_rate

class LogMelSpectrogram(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.melspctrogram = transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=h.n_fft,
            win_length=h.win_size,
            hop_length=h.hop_size,
            center=False,   # 啥
            power=1.0,
            norm="slaney",  # 啥
            onesided=True,  # 啥
            n_mels=h.num_mels,
            mel_scale="slaney", # 啥
            f_min=0,
            f_max=8000,
        )

    def forward(self, wav):
        wav = F.pad(wav, ((1024 - 160) // 2, (1024 - 160) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5)) # 好像知道是啥
        return logmel