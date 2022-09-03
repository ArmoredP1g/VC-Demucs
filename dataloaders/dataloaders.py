from configs import mel_cfg
from configs import training_cfg
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import pandas as pd
from torchaudio.functional import resample
from model.wav2mel import LogMelSpectrogram


class MultiUtteranceData(Dataset):
    '''
        dataloader for training voice2vec
    '''
    def __init__(self, sample_rate, data_path, uttr):
        super().__init__()
        self.path = data_path
        self.sample_rate = sample_rate
        self.data = pd.read_csv(data_path + "\\utterance_info.csv")
        self.speakers = self.data.speaker_id.unique()
        self.uttr = uttr
        self.log_mel_spec = LogMelSpectrogram(
                                mel_cfg.sampling_rate,
                                mel_cfg.n_fft,
                                mel_cfg.win_size,
                                mel_cfg.hop_size,
                                mel_cfg.num_mels
                            )

    def __getitem__(self, index):
        # Pick audio randomly, and padding all these to the longest one.
        wav_files = self.data[self.data['speaker_id'] == self.speakers[index]].sample(self.uttr, replace=True)["utterance"].to_list()
        mel_list = []
        for path in wav_files:
            pth = self.path + '/' + self.speakers[index] + path
            wav_tensor, sr = torchaudio.load(pth)

            # if sample rate doesnt match, resample it
            if sr != self.sample_rate:
                wav_tensor = resample(wav_tensor, sr, self.sample_rate)
            
            _, logmel = self.log_mel_spec(wav_tensor)   # b,mel,lenth
            mel_list.append(logmel[0].T)
        
        return pad_sequence(mel_list, batch_first=True)


    def __len__(self):
        return self.speakers.__len__()