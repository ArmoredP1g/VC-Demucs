from configs import mel_cfg
from configs import training_cfg
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from random import randint
import torchaudio
import pandas as pd
from torchaudio.functional import resample
from model.wav2mel import LogMelSpectrogram


class MultiUtteranceData_mel(Dataset):
    '''
        dataloader for training voice2vec
    '''
    def __init__(self, sample_rate, data_path, uttr_num, clip_size):
        super().__init__()
        self.path = data_path
        self.sample_rate = sample_rate
        self.data = pd.read_csv(data_path + "\\utterance_info.csv")
        self.speakers = self.data.speaker_id.unique()
        self.uttr_num = uttr_num
        self.clip_size = clip_size
        self.log_mel_spec = LogMelSpectrogram(
                                sample_rate,
                                1024,
                                186*4,
                                186,
                                80
                            )

    def __getitem__(self, index):
        # Pick audio randomly, and padding all these to the longest one.
        wav_files = self.data[self.data['speaker_id'] == self.speakers[index]].sample(self.uttr_num, replace=True)["utterance"].to_list()
        mel_list = []
        for path in wav_files:
            pth = self.path + '/' + self.speakers[index] + path
            wav_tensor, sr = torchaudio.load(pth)

            # if sample rate doesnt match, resample it
            if sr != self.sample_rate:
                wav_tensor = resample(wav_tensor, sr, self.sample_rate)
            _, logmel = self.log_mel_spec(wav_tensor)   # b(1),mel,lenth

            # taking 3s clips
            if logmel.shape[2] > self.clip_size:
                upper_idx = logmel.shape[2] - (logmel.shape[2]%self.clip_size) - 1
                random_idx = randint(0,upper_idx)
                logmel = logmel[:,:,random_idx:random_idx+self.clip_size]

            mel_list.append(logmel[0].T)
        
        # return pad_sequence(mel_list, batch_first=True).permute(0,2,1)
        return pad_sequence(mel_list, batch_first=True)  # Permute is not needed here, but in batch_padding
                                                         # return: uttr_num, lenth, mel_num

    def __len__(self):
        return self.speakers.__len__()


class MultiUtteranceData_raw(Dataset):
    '''
        dataloader for training voice2vec
    '''
    def __init__(self, sample_rate, data_path, uttr_num, clip_size):
        super().__init__()
        self.path = data_path
        self.sample_rate = sample_rate
        self.data = pd.read_csv(data_path + "\\utterance_info.csv")
        self.speakers = self.data.speaker_id.unique()
        self.uttr_num = uttr_num
        self.clip_size = clip_size

    def __getitem__(self, index):
        # Pick audio randomly, and padding all these to the longest one.
        wav_files = self.data[self.data['speaker_id'] == self.speakers[index]].sample(self.uttr_num, replace=True)["utterance"].to_list()
        wave_list = []
        for path in wav_files:
            pth = self.path + '/' + self.speakers[index] + path
            wav_tensor, sr = torchaudio.load(pth)

            # if sample rate doesnt match, resample it
            if sr != self.sample_rate:
                wav_tensor = resample(wav_tensor, sr, self.sample_rate)

            # taking 3s clips
            if wav_tensor.shape[1] > self.clip_size:
                upper_idx = wav_tensor.shape[1] - (wav_tensor.shape[1]%self.clip_size) - 1
                random_idx = randint(0,upper_idx)
                wav_tensor = wav_tensor[:,random_idx:random_idx+self.clip_size]

            wave_list.append(wav_tensor[0])
        
        return pad_sequence(wave_list, batch_first=True)  # return: uttr_num, lenth
        
    def __len__(self):
        return self.speakers.__len__()


def batch_padding(data):
    l = []
    for d in data:
        # d: [uttr_num, lenth, mel_num]
        l.append(d.permute(1,0,2))

    return pad_sequence(l, batch_first=True).permute(0,2,3,1)    # [b,len,uttr,mel] ----> [b,uttr,mel,len]