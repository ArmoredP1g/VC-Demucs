

if __name__ == "__main__":
    import json
    import shutil
    from torch.utils.tensorboard import SummaryWriter
    from dataloaders.dataloaders import MultiUtteranceData, batch_padding
    from torch.utils.data import DataLoader
    from model.Voice2Vec import Voice2Vec, simplified_ge2e_loss
    from torch.optim import Adam
    import torch
    from torch.cuda.amp import autocast as autocast
    from configs.training_cfg import device
    import os

    # args
    ckpt_path = "ckpt/multiplier8_25000"
    use_tensorboard = True
    tensorboard_path = ""
    tag = "multiplier8"
    dataset_path = "E:/vox2_converted"
    batch_size = 4
    num_worker = 8
    loss_log_interval = 200
    ckpt_save_interval = 5000


    total_step = 0
    total_loss = 0

    sum_writer = None
    dataset = MultiUtteranceData(22050, dataset_path, batch_size)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_worker, drop_last=True, collate_fn=batch_padding)

    v2v_model = Voice2Vec().to("cuda")

    if ckpt_path != "":
        # 读取 ckpt
        v2v_model.load_state_dict(torch.load(ckpt_path + "/state_dict.pt"))
        # 读取训练参数
        f = open(ckpt_path + '/arc.json', 'r')
        content = f.read()
        total_step = json.loads(content)["total_step"]
    else:
        # 新模型
        v2v_model.weight_init() # @todo

    if tensorboard_path == "":
        tensorboard_path = "./runs"

    if use_tensorboard:
        sum_writer = SummaryWriter(tensorboard_path+"/{}".format(tag))

    # for dim-reduction visualization
    _, sample_wav = enumerate(dataloader, 0).__next__()
    sample_wav = sample_wav.reshape(batch_size*batch_size, sample_wav.shape[2], sample_wav.shape[3]).to(device)
    meta = []
    for i in range(batch_size):
        meta += [i]*batch_size

    optimizer = Adam(v2v_model.parameters(),
                    lr=0.001,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=0,
                    amsgrad=False)


    while True:
        for index,data in enumerate(dataloader, 0):
            total_step += 1
            optimizer.zero_grad()

            spk, uttr, mel, l = data.shape

            # with autocast():
            output = v2v_model(data.reshape(spk*uttr, mel, l).to(device))
            output = output.view(spk, uttr, 512)
            center_loss, cross_loss = simplified_ge2e_loss(output)
            loss = center_loss + cross_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if total_step%50 == 0:
                print("center_loss: {} ,cross_loss: {}".format(center_loss, cross_loss))

            if total_step%loss_log_interval == 0:
                # print("total step: {},  loss: {}".format(total_step, total_loss/loss_log_interval))

                if use_tensorboard:
                    sum_writer.add_scalar(tag='Loss',
                                            scalar_value=total_loss/loss_log_interval,
                                            global_step=total_step
                                        )
                
                total_loss = 0

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

                torch.save(v2v_model.state_dict(),p + "/state_dict.pt")
                f = open(p + '/arc.json', 'w')
                f.write(j)
                f.close()

                if use_tensorboard:
                    emb = v2v_model(sample_wav)
                    sum_writer.add_embedding(
                                            mat=emb,
                                            metadata=meta,
                                            global_step=total_step
                                        )  


        