

if __name__ == "__main__":
    from dataloaders.dataloaders import MultiUtteranceData, batch_padding
    from torch.utils.data import DataLoader
    from model.Voice2Vec import Voice2Vec, simplified_ge2e_loss
    from torch.optim import Adam
    import torch
    from torch.cuda.amp import autocast as autocast
    from configs.training_cfg import device

    dataset = MultiUtteranceData(22050, "E:/vox2_converted", 4)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4, num_workers=8, drop_last=True, collate_fn=batch_padding)

    v2v_model = Voice2Vec().to("cuda")
    v2v_model.weight_init()

    optimizer = Adam(v2v_model.parameters(),
                    lr=0.001,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=0,
                    amsgrad=False)

    
    total_step = 0

    while True:
        for index,data in enumerate(dataloader, 0):
            total_step += 1
            optimizer.zero_grad()

            spk, uttr, mel, l = data.shape

            # with autocast():
            output = v2v_model(data.reshape(spk*uttr, mel, l).to(device))
            output = output.view(spk, uttr, 512)
            loss = simplified_ge2e_loss(output)

            loss.backward()
            optimizer.step()

            if total_step%200 == 0:
                print("total step: {},  loss: {}".format(total_step, loss))

            if total_step%2500 == 0:
                torch.save(v2v_model.state_dict(),"test_{}.pt".format(total_step))