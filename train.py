import pandas as pd
import torch
from setting import get_config
import os
from utils.builder import build_dataloader, build_linear, build_sup
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.trainer import train_linear, val_linear, train_step, val_step
from utils.tfunction import save_model
import logging
import numpy as np

if __name__ == "__main__":
    # load args
    args, writer = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.training["gpu_id"]
    torch.manual_seed(args.training["seed"])
    backbone, linear = args.model["backbone"], args.model["linear"]
    num_classes, ckpt = args.model["num_classes"], args.ckpt
    batch_size, num_workers, epochs = args.training["batch_size"], args.training["num_workers"], args.training["epochs"]
    lr, momentum, weight_decay = args.optimizer["lr"], args.optimizer["momentum"], args.optimizer["weight_decay"]
    T_max, eta_min = args.scheduler["T_max"], args.scheduler["eta_min"]
    csv_path = args.path[args.model["dataset"]]["csv_path"]
    img_path = args.path[args.model["dataset"]]["img_path"]
    pth_save_path = args.path["pth"]
    save_freq = args.training["save_freq"]

    for fold in range(5):
        if fold < 10:
            logging.info(f"---------------fold:{fold}---------------")
            df_train = pd.read_csv(f"{csv_path}/{fold}train.csv")
            df_val = pd.read_csv(f"{csv_path}/{fold}val.csv")
            images_train = [img_path + i for i in df_train["modal1_path_list"].tolist()]
            images_val = [img_path + i for i in df_val["modal1_path_list"].tolist()]
            labels_train = df_train["labels"].tolist()
            labels_val = df_val["labels"].tolist()

            # build dataloader&model
            set = "linear" if linear else "sup"
            dataloader_train = build_dataloader(images_train, labels_train, batch_size, num_workers, set)
            dataloader_val = build_dataloader(images_val, labels_val, 1, num_workers, "val")
            if linear:
                print(ckpt[fold], os.path.exists(ckpt[fold]))
                model, classifier, criterion = build_linear(backbone, ckpt[fold], num_classes)
                optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                model, criterion = build_sup(backbone, ckpt[fold], num_classes)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

            # training routine
            results_list = []
            best_acc = 0.9
            for epoch in range(1, epochs + 1):
                # train for one epoch
                if linear:
                    loss_train, acc_train = train_linear(
                        dataloader_train, model, classifier, criterion, optimizer, epoch
                    )
                    loss_val, acc_val, auc_val, recall, recall0, pre, pre0 = val_linear(
                        dataloader_val, model, classifier, criterion
                    )
                    if epoch % save_freq == 0:
                        save_file = f"{pth_save_path}/{fold}epoch{epoch}.pth"
                        save_model(classifier, optimizer, epoch, save_file)
                    if acc_val > best_acc:
                        save_file = f"{pth_save_path}/{fold}best{epoch}.pth"
                        best_acc = acc_val
                        save_model(classifier, optimizer, epoch, save_file)
                else:
                    loss_train, acc_train = train_step(dataloader_train, model, criterion, optimizer, epoch)
                    loss_val, acc_val, auc_val, recall, recall0, pre, pre0 = val_step(dataloader_val, model, criterion)

                    if epoch % save_freq == 0:
                        save_file = f"{pth_save_path}/{fold}epoch{epoch}.pth"
                        save_model(model, optimizer, epoch, save_file)
                    if acc_val > best_acc:
                        save_file = f"{pth_save_path}/{fold}best{epoch}.pth"
                        best_acc = acc_val
                        save_model(model, optimizer, epoch, save_file)

                scheduler.step()

                writer.add_scalar(f"{fold}/loss_train", loss_train, epoch)
                writer.add_scalar(f"{fold}/acc_train", acc_train, epoch)
                writer.add_scalar(f"{fold}/loss_val", loss_val, epoch)
                writer.add_scalar(f"{fold}/acc_val", acc_val, epoch)
                writer.add_scalar(f"{fold}/auc_val", auc_val, epoch)

                results = [acc_val, auc_val, recall, recall0, pre, pre0]
                results_list.append(results)
                df = pd.DataFrame(
                    {
                        "acc_val": list(np.array(results_list).T[0]),
                        "auc_val": list(np.array(results_list).T[1]),
                        "recall": list(np.array(results_list).T[2]),
                        "recall0": list(np.array(results_list).T[3]),
                        "pre": list(np.array(results_list).T[4]),
                        "pre0": list(np.array(results_list).T[5]),
                    }
                )
                df.to_csv(args.path["result"] + "/" + str(fold) + ".csv")
