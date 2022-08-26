"""
Convert the source audio of the VOX2 dataset to WAV format
pydub & pandas needed
"""

from ast import arg
from email.mime import audio
from ntpath import join
import os 
import re
import pandas as pd
import argparse
from pydub import AudioSegment

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", required=True, help="root dir of vox dataset")
parser.add_argument("-t", "--target", required=True, help="dir of converted dataset")
parser.add_argument("-br", "--bitrate", help="sample rate", default=16000)

args = vars(parser.parse_args())
dir = args["dir"]
target = args["target"]
br = args["bitrate"]

os.mkdir(target)

info_list = []


g = os.walk(dir)  
for path,dir_list,file_list in g:
    for file_name in file_list:  
        if re.search(r'\.m4a$', file_name):
            info = {}
            info["speaker_id"] = re.search(r'id\d{5}', path).group()
            info["utterance"] = '/'+file_name.split('.')[0]+".wav"

            if not os.path.exists(target+info["speaker_id"]):
                os.mkdir(target+info["speaker_id"])

            audio = AudioSegment.from_file(join(path, file_name))
            audio.export(join(target,info["speaker_id"]+"\\"+info["utterance"]+".wav"), format='wav', bitrate=br)
            
            info_list.append(info)

# save csv
df = pd.DataFrame(info_list, columns=["speaker_id", "utterance"])
df.to_csv(target+"utterance_info.csv")
print("--done--")