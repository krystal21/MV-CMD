import pandas as pd
import torch
import numpy as np
from setting import get_config
import os
import logging
from utils.builder import build_dataloader, build_model
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.trainer import train_viewcon
from utils.tfunction import save_model


if __name__ == "__main__":
    # load args
    args, writer = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.training["gpu_id"]
    torch.manual_seed(args.training["seed"])
    backbone = args.model["backbone"]
    batch_size, num_workers, epochs = args.training["batch_size"], args.training["num_workers"], args.training["epochs"]
    t = args.training["t"]
    save_freq = args.training["save_freq"]
    lr, momentum, weight_decay = args.optimizer["lr"], args.optimizer["momentum"], args.optimizer["weight_decay"]
    T_max, eta_min = args.scheduler["T_max"], args.scheduler["eta_min"]
    pth_save_path = args.path["pth"]
    csv_path = args.path[args.model["dataset"]]["csv_path"]
    img_path = args.path[args.model["dataset"]]["img_path"]

    for fold in range(5):
        if fold < 6:
            logging.info(f"---------------fold:{fold}---------------")
            df_train = pd.read_csv(f"{csv_path}/{fold}train.csv")
            # df_val = pd.read_csv(f"{csv_path}/{fold}val.csv")
            img_list = [img_path + i for i in df_train["modal1_path_list"].tolist()]
            img_list1 = [img_path + i for i in df_train["modal2_path_list"].tolist()]
            labels = df_train["labels"].tolist()
            # build dataloader&model
            dataloader = build_dataloader(img_list, labels, batch_size, num_workers, "viewcon", img_list1)
            model, criterion = build_model(backbone, t)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

            # training routine
            for epoch in range(1, epochs + 1):
                # train for one epoch
                loss, loss_weight = train_viewcon(dataloader, model, criterion, optimizer, epoch)
                writer.add_scalar(f"{fold}/loss", loss, epoch)
                writer.add_scalar(f"{fold}/loss_weight", loss_weight, epoch)
                writer.add_scalar(f"{fold}/lr", optimizer.param_groups[0]["lr"], epoch)
                if epoch % save_freq == 0:
                    save_file = f"{pth_save_path}/{fold}epoch{epoch}.pth"
                    save_model(model, optimizer, epoch, save_file)

                scheduler.step()
            # save the last model
            save_file = f"{pth_save_path}/{fold}last.pth"
            save_model(model, optimizer, epochs, save_file)
