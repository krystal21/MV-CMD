# -*- coding: utf-8 -*-
"""
时间：2023年06月06日
"""
import logging
import argparse
import sys
import yaml
import os
import datetime
import time
from tensorboardX import SummaryWriter


def merge_config(config, args):
    # 更新参数值
    if args.dataset is not None:
        config["model"]["dataset"] = args.dataset
    if args.save_dir is not None:
        config["model"]["save_dir"] = args.save_dir
    if args.gpu_id is not None:
        config["training"]["gpu_id"] = args.gpu_id
    if args.seed_torch is not None:
        config["training"]["seed"] = args.seed_torch
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.T is not None:
        config["kd"]["kd_T"] = args.T
    if args.distill is not None:
        config["kd"]["distill"] = args.distill
    if args.co is not None:
        config["kd"]["coefficient"] = args.co
    return config


def get_config():
    parser = argparse.ArgumentParser("Deep learning parameters")
    parser.add_argument("--config", help="configuration file", type=str, default="config/config.yml")
    parser.add_argument("--save_dir", type=str, help="save exp floder name")
    parser.add_argument("--gpu_id", type=str, help="Gpu id lists")
    parser.add_argument("--mod", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--seed_torch", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--T", type=float)
    parser.add_argument("--distill", type=str)
    parser.add_argument("--co", nargs="+", type=int, help="A list of values")

    args = parser.parse_args()
    # 读取YAML配置文件
    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        # 合并配置exp1
        config = merge_config(yaml_config, args)
        args = argparse.Namespace(**config)

    args.exp_name = args.model["save_dir"] + "-" + datetime.datetime.now().strftime("%Y-%m-%d %H-%M")
    exp = args.model["exp"]
    # 文件处理
    if not os.path.exists(os.path.join(exp, args.exp_name)):
        os.makedirs(os.path.join(exp, args.exp_name))

    if not os.path.exists(os.path.join(args.model["save_dir"], args.exp_name, "result")):
        os.makedirs(os.path.join(exp, args.exp_name, "result"))
    if not os.path.exists(os.path.join(exp, args.exp_name, "pth")):
        os.makedirs(os.path.join(exp, args.exp_name, "pth"))

    args.path["result"] = os.path.join(exp, args.exp_name, "result")
    args.path["pth"] = os.path.join(exp, args.exp_name, "pth")

    # 日志文件
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")

    fh = logging.FileHandler(os.path.join(exp, args.exp_name, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)
    # 配置文件
    with open(os.path.join(exp, args.exp_name, "config.yml"), "w") as f:
        yaml.dump(args, f)

    # Tensorboard文件
    writer = SummaryWriter(f'{exp}/{args.exp_name}/runs/{time.strftime("%Y-%m-%d_%H-%M", time.localtime())}')
    return args, writer


if __name__ == "__main__":
    args, writer = get_config()
