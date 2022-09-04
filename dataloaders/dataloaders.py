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
    def __init__(self, sample_rate, data_path, uttr_num):
        super().__init__()
        self.path = data_path
        self.sample_rate = sample_rate
        self.data = pd.read_csv(data_path + "\\utterance_info.csv")
        self.speakers = self.data.speaker_id.unique()
        self.uttr_num = uttr_num
        self.log_mel_spec = LogMelSpectrogram(
                                sample_rate,
                                1024,
                                1024,
                                256,
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
            mel_list.append(logmel[0].T)
        
        # return pad_sequence(mel_list, batch_first=True).permute(0,2,1)
        return pad_sequence(mel_list, batch_first=True)  # Permute is not needed here, but in batch_padding
                                                         # return: uttr_num, lenth, mel_num


    def __len__(self):
        return self.speakers.__len__()


def batch_padding(data):
    l = []
    for d in data:
        # d: [uttr_num, lenth, mel_num]
        l.append(d.permute(1,0,2))

    return pad_sequence(l, batch_first=True).permute(0,2,3,1)    # [b,len,uttr,mel] ----> [b,uttr,mel,len]