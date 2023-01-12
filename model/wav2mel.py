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
    def __init__(self, **kwargs):

        self.args = {
            'sample_rate': 22050,
            'n_fft':2048,              #  creates n_fft // 2 + 1 bins.
            'win_length':1024,         #  default = n_fft
            'hop_length':512,         #  default = win_length // 2
            'n_mels':256,
            'normalized':False,
            'norm':'slaney',
            'mel_scale':'slaney'      # librosa mel_to_audio
        }

        self.args.update(kwargs)

        super().__init__()
        self.melspctrogram = transforms.MelSpectrogram(
                sample_rate=self.args['sample_rate'],
                n_fft=self.args['n_fft'],              #  creates n_fft // 2 + 1 bins.
                win_length=self.args['win_length'],         #  default = n_fft
                hop_length=self.args['hop_length'],         #  default = win_length // 2
                n_mels=self.args['n_mels'],
                normalized=self.args['normalized'],
                norm=self.args['norm'],
                mel_scale=self.args['mel_scale']      # librosa mel_to_audio
        )

    def forward(self, wav):
        # wav = F.pad(wav, ((1024 - 160) // 2, (1024 - 160) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5)) # 好像知道是啥
        return mel, logmel