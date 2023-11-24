import pandas as pd
import torch
from utils.trainer import train_kd, val_stu, train_step, val_step
import numpy as np
from setting import get_config
import os
import logging
from distiller_zoo import DistillKL, HintLoss
import torch.nn as nn
from utils.builder import build_dataloader, build_kd, build_memory_bank, build_memory_bank_linear
from utils.model import get_model
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn
from utils.tfunction import save_model
from PIL import Image
from torchvision import transforms


if __name__ == "__main__":
    # 加载参数
    args, writer = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.training["gpu_id"]
    torch.manual_seed(args.training["seed"])
    backbone, linear = args.model["backbone"], args.model["linear"]
    num_classes, ckpt, linear_ckpt = args.model["num_classes"], args.ckpt, args.linear_ckpt
    batch_size, num_workers, epochs = args.training["batch_size"], args.training["num_workers"], args.training["epochs"]
    lr, momentum, weight_decay = args.optimizer["lr"], args.optimizer["momentum"], args.optimizer["weight_decay"]
    T_max, eta_min = args.scheduler["T_max"], args.scheduler["eta_min"]
    csv_path = args.path[args.model["dataset"]]["csv_path"]
    img_path = args.path[args.model["dataset"]]["img_path"]
    distill, kd_T, coefficient = args.kd["distill"], args.kd["kd_T"], args.kd["coefficient"]
    pth_save_path = args.path["pth"]
    result_path = args.path["result"]
    ifviewcon = args.kd["viewcon"]

    transform_list_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    for fold in range(5):
        logging.info(f"---------------fold:{fold}---------------")
        df_train = pd.read_csv(f"{csv_path}/wle/{fold}train.csv")
        df_val = pd.read_csv(f"{csv_path}/wle/{fold}val.csv")
        images_train = [img_path + i for i in df_train["modal1_path_list"].tolist()]
        images_val = [img_path + i for i in df_val["modal1_path_list"].tolist()]
        labels_train = df_train["labels"].tolist()
        labels_val = df_val["labels"].tolist()
        images_tea = [img_path + i for i in df_train["modal2_path_list"].tolist()]
        images_tea_val = [img_path + i for i in df_val["modal2_path_list"].tolist()]
        img_list = df_train["img_list"].tolist()

        dataloader_train = build_dataloader(
            images_train, labels_train, batch_size, num_workers, "kd", images_tea, img_list
        )
        dataloader_val = build_dataloader(images_val, labels_val, 1, num_workers, "val")
        dataloader_val_tea = build_dataloader(images_tea_val, labels_val, 1, num_workers, "val")

        # 模型
        print(distill, ckpt[fold])
        # memory bank
        df_m = pd.read_csv(f"{csv_path}/nbi/{fold}train.csv")
        images_nbi = df_m["modal1_path_list"].tolist()
        if ifviewcon:
            model_t, model_s, classifier = build_kd(backbone, ckpt[fold], num_classes, ifviewcon, linear_ckpt[fold])
            model_t.cuda()
            classifier.cuda()
            memory_bank = build_memory_bank_linear(images_nbi, model_t, classifier, img_path)
        else:
            model_t, model_s = build_kd(backbone, ckpt[fold], num_classes, ifviewcon, linear_ckpt[fold])
            model_t.cuda()
            memory_bank = build_memory_bank(images_nbi, model_t, img_path)
        module_list = nn.ModuleList([])
        module_list.append(model_s)
        trainable_list = nn.ModuleList([])
        trainable_list.append(model_s)

        if "hint" in distill:
            raw = False
            criterion_kd = HintLoss(2048, 2048, False, raw)
            if not raw:
                module_list.append(criterion_kd.embed_s)
                trainable_list.append(criterion_kd.embed_s)
        else:
            criterion_kd = DistillKL(kd_T)

        # criterion
        criterion_cls = nn.CrossEntropyLoss()
        criterion_div = DistillKL(kd_T)
        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)  # classification loss
        criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
        criterion_list.append(criterion_kd)  # other knowledge distillation loss
        criterion_list.cuda()

        optimizer = torch.optim.Adam(trainable_list.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        if ifviewcon:
            module_list.append(classifier)
        module_list.append(model_t)
        if torch.cuda.is_available():
            module_list.cuda()
            criterion_list.cuda()
            cudnn.benchmark = True

        # train
        list_train = []
        best_acc = 0.8
        for epoch in range(epochs):
            logging.info(f"epoch:{epoch}")

            epoch_loss_train, cls_loss, kd_loss, div_loss = train_kd(
                dataloader_train,
                module_list,
                criterion_list,
                optimizer,
                distill,
                coefficient,
                linear=ifviewcon,
                memory_bank=memory_bank,
            )
            model_s = module_list[0]
            epoch_loss_val, acc_val, auc_val, recall_val, recall0_val, pre_val, pre0_val = val_stu(
                dataloader_val, model_s, criterion_cls
            )
            scheduler.step()
            writer.add_scalar(f"{fold}/loss_all", epoch_loss_train, epoch)
            writer.add_scalar(f"{fold}/loss_train", cls_loss, epoch)
            writer.add_scalar(f"{fold}/kd_loss", kd_loss, epoch)
            writer.add_scalar(f"{fold}/div_loss", div_loss, epoch)
            writer.add_scalar(f"{fold}/loss_val", epoch_loss_val, epoch)
            writer.add_scalar(f"{fold}/acc_val", acc_val, epoch)
            writer.add_scalar(f"{fold}/auc_val", auc_val, epoch)
            # writer.add_scalar(f"{fold}/val/recall", recall_val, epoch)
            # writer.add_scalar(f"{fold}/val/recall0", recall0_val, epoch)
            # writer.add_scalar(f"{fold}/val/pre", pre_val, epoch)
            # writer.add_scalar(f"{fold}/val/pre0", pre0_val, epoch)

            result_train = [acc_val, auc_val, recall_val, recall0_val, pre_val, pre0_val]
            list_train.append(result_train)

            if acc_val > best_acc:
                save_file = f"{pth_save_path}/{fold}best{epoch}.pth"
                best_acc = acc_val
                save_model(model_s, optimizer, epoch, save_file)

            df = pd.DataFrame(
                {
                    "acc_val": list(np.array(list_train).T[0]),
                    "auc_val": list(np.array(list_train).T[1]),
                    "recall_val": list(np.array(list_train).T[2]),
                    "recall0_val": list(np.array(list_train).T[3]),
                    "pre_val": list(np.array(list_train).T[4]),
                    "pre_val0": list(np.array(list_train).T[5]),
                }
            )
            df.to_csv(result_path + "/" + str(fold) + ".csv")
