if __name__ == "__main__":
    import json
    import shutil
    from torch.utils.tensorboard import SummaryWriter
    from dataloaders.dataloaders import UtteranceData_mel, batch_padding
    from torch.utils.data import DataLoader
    from torch.nn.utils import clip_grad_norm_
    from model.VAE_VC import Content_Encoder, Speaker_Encoder, Reparameterize
    from model.WaveNet import Conditional_WaveNet
    from model.wav2mel import LogMelSpectrogram
    from torch.optim import Adam
    import torch
    import torchaudio
    from torchaudio.functional import resample
    from librosa.feature.inverse import mel_to_audio
    from torch.cuda.amp import autocast as autocast
    from configs.training_cfg import device, ce_args, se_args
    import os
    from torch import autograd

    torch.autograd.set_detect_anomaly(True)

    # args
    ckpt_path = ""
    use_tensorboard = True
    tensorboard_path = ""
    tag = "adain_wavenet2"
    dataset_path = "D:/vox2_converted"
    batch_size = 16
    num_worker = 4
    loss_log_interval = 200
    ckpt_save_interval = 5000
    demo_interval = 5000


    total_step = 0
    total_loss = 0

    sum_writer = None
    dataset = UtteranceData_mel(22050, dataset_path, 500)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_worker, drop_last=True, collate_fn=batch_padding)

    content_encoder = Content_Encoder(**ce_args).to("cuda")
    speaker_encoder = Speaker_Encoder(**se_args).to("cuda")
    decoder = Conditional_WaveNet(128,256,196,128,5,5).to("cuda")

    # 效果可视化
    log_mel_spec = LogMelSpectrogram()
    wav_content, sr1 = torchaudio.load("D:\\vox2_converted\\id00021\\00015.wav", normalize=True)
    wav_timbre, sr2 = torchaudio.load("D:\\vox2_converted\\id00022\\00028.wav", normalize=True)
    wav_content = resample(wav_content, sr1, 22050)
    wav_timbre = resample(wav_timbre, sr2, 22050)
    mel_content, _ = log_mel_spec(wav_content)
    mel_timbre, _ = log_mel_spec(wav_timbre)
    del wav_content
    del wav_timbre

    if ckpt_path != "":
        # 读取 ckpt
        content_encoder.load_state_dict(torch.load(ckpt_path + "/content_encoder.pt"))
        speaker_encoder.load_state_dict(torch.load(ckpt_path + "/speaker_encoder.pt"))
        decoder.load_state_dict(torch.load(ckpt_path + "/decoder.pt"))
        # 读取训练参数
        f = open(ckpt_path + '/arc.json', 'r')
        content = f.read()
        total_step = json.loads(content)["total_step"]
    else:
        # 新模型
        # v2v_model.weight_init() # @todo
        pass

    if tensorboard_path == "":
        tensorboard_path = "./runs"

    if use_tensorboard:
        sum_writer = SummaryWriter(tensorboard_path+"/{}".format(tag))

    optimizer = Adam([
                {'params': content_encoder.parameters(), 'lr': 0.0005}, 
                {'params': speaker_encoder.parameters(), 'lr': 0.001},
                {'params': decoder.parameters(), 'lr': 0.001}
            ])

    l1_loss = torch.nn.L1Loss()


    while True:
        for index,data in enumerate(dataloader, 0):
            total_step += 1
            print(total_step)
            optimizer.zero_grad()

            data = data.transpose(1,2).to(device)
            spk, mel, len = data.shape
            # with autocast():  #会导致nan，原因不明
            miu, logvar = content_encoder(data)
            content = Reparameterize(miu, logvar)
            gamma, beta = speaker_encoder(data)
            # output_mel = decoder(content, gamma, beta)
            output_mel = decoder(content, None, None)

            reconstruction_loss = 10*l1_loss(output_mel.transpose(1,2), data.transpose(1,2))
            kl_loss = 0.01*(0.5 * torch.mean(torch.exp(logvar) + miu**2 - 1. - logvar))
            
            (reconstruction_loss + kl_loss).backward()
            clip_grad_norm_(content_encoder.parameters(), 5)
            clip_grad_norm_(speaker_encoder.parameters(), 5)
            clip_grad_norm_(decoder.parameters(), 5)
            optimizer.step()

            total_loss += (reconstruction_loss + kl_loss).item()

            if total_step%50 == 0:
                print("reconstruction_loss: {} ,kl_loss: {}".format(reconstruction_loss, kl_loss))

            if total_step%loss_log_interval == 0:
                if use_tensorboard:
                    sum_writer.add_scalar(tag='Loss',
                                            scalar_value=total_loss/loss_log_interval,
                                            global_step=total_step
                                        )
                total_loss = 0

            if total_step%demo_interval == 0:
                if use_tensorboard:
                    with torch.no_grad():
                        g, b = speaker_encoder(mel_timbre.to(device))
                        c, _ = content_encoder(mel_content.to(device))  #miu only
                        o = torch.clamp(decoder(c, g, b), min=0)
                        owav = mel_to_audio(o.cpu().numpy(), sr=22050, n_fft=2048, hop_length=512, win_length=1024, n_iter=500, norm='slaney')

                        sum_writer.add_audio(tag='转换效果',
                                            snd_tensor=owav,
                                            global_step=total_step,
                                            sample_rate=22050
                                            )

            if total_step%ckpt_save_interval == 0:

                p = "./ckpt/{}_{}".format(tag, total_step)
                if not os.path.exists(p):
                    os.makedirs(p)
                else:
                    shutil.rmtree(p)  
                    os.mkdir(p)

                j = json.dumps({
                    'total_step':total_step,
                })

                torch.save(content_encoder.state_dict(),p + "/content_encoder.pt")
                torch.save(speaker_encoder.state_dict(),p + "/speaker_encoder.pt")
                torch.save(decoder.state_dict(),p + "/decoder.pt")
                f = open(p + '/arc.json', 'w')
                f.write(j)
                f.close()


        