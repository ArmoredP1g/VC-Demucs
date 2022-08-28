"""
Convert the source audio of the VOX2 dataset to WAV format
pydub & pandas needed
"""

from ntpath import join
import os
import re
from pydub import AudioSegment

def Sub_process(q):
    while True:
        end, origin_path, target_path, sr = q.get()
        if not end:
            audio = AudioSegment.from_file(file=origin_path)
            audio.frame_rate = sr
            audio.export(target_path, format='wav')
        else:
            break

if __name__ == "__main__":
    import pandas as pd
    import argparse
    from multiprocessing import Process
    from multiprocessing import Queue

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", required=True, help="root dir of vox dataset")
    parser.add_argument("-t", "--target", required=True, help="dir of converted dataset")
    parser.add_argument("--sr", help="sample rate", default=22050)
    parser.add_argument("-p", "--process", help="Num of processes", default=4)

    args = vars(parser.parse_args())
    dir = args["dir"]
    target = args["target"]
    sr = args["sr"]
    process = int(args["process"])

    os.mkdir(target)

    q = Queue(300)
    process_list = []
    for i in range(process):
        p = Process(target=Sub_process,args=(q,))
        p.start()
        process_list.append(p)

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

                q.put((False, join(path, file_name), join(target,info["speaker_id"]+"\\"+info["utterance"]), sr))
                
                info_list.append(info)


    # notify all threads to quit..
    for i in range(process):
        q.put((True, "", "", 0))

    for p in process_list:
        p.join()

    # save csv
    df = pd.DataFrame(info_list, columns=["speaker_id", "utterance"])
    df.to_csv(target+"utterance_info.csv")
    
    print("--done--")