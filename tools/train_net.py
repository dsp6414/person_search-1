import argparse
import logging
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import _init_paths  # noqa: F401
from datasets.psdb import PSDB
from datasets.sampler import PSSampler
from models.network import Network
from utils.config import cfg, cfg_from_file
from utils.utils import init_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", help="Config file. Default: None")
    parser.add_argument(
        "--gpu", default=0, type=int, help="GPU device id to use, -1 for CPU. Default: 0"
    )
    parser.add_argument(
        "--checkpoint", help="Resume at a specified checkpoint. Default: None",
    )
    parser.add_argument("--tbX", action="store_true", help="Enable tensorboardX. Default: False")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    init_logger("train.log")
    logging.info("Called with args:\n" + str(args))

    if args.cfg:
        cfg_from_file(args.cfg)
    if args.tbX:
        # Use tensorboardX to visualize experimental results
        from tensorboardX import SummaryWriter

        tb_log_path = osp.join(cfg.DATA_DIR, "tb_logs")
        logger = SummaryWriter(tb_log_path)

    # Fix the random seeds (numpy and pytorch) for reproducibility.
    if cfg.RANDOM_SEED != -1:
        logging.info("Fix random seed to %s." % cfg.RANDOM_SEED)
        torch.manual_seed(cfg.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.RANDOM_SEED)
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(cfg.RANDOM_SEED)
        np.random.seed(cfg.RANDOM_SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.RANDOM_SEED)

    # Initialize dataset
    dataset = PSDB("psdb_train")
    dataloader = DataLoader(dataset, batch_size=1, sampler=PSSampler(dataset))
    data_iter = iter(dataloader)
    logging.info("Loaded dataset: psdb_train")

    # Initialize model
    net = Network()
    net.train()
    device = torch.device("cuda:%s" % args.gpu if args.gpu != -1 else "cpu")
    net.to(device)
    if not args.checkpoint:
        pretrained_model = osp.join(cfg.DATA_DIR, "pretrained_model", "resnet50_caffe.pth")
        net.load_state_dict(torch.load(pretrained_model))
        logging.info("Loaded pretrained model from: %s" % pretrained_model)

    # Initialize optimizer
    lr = cfg.TRAIN.LEARNING_RATE
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    params = []
    for k, v in net.named_parameters():
        if v.requires_grad:
            if "BN" in k:
                params += [{"params": [v], "lr": lr, "weight_decay": 0}]
            elif "bias" in k:
                params += [{"params": [v], "lr": 2 * lr, "weight_decay": 0}]
            else:
                params += [{"params": [v], "lr": lr, "weight_decay": weight_decay}]
    optimizer = optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    # --------------------
    # Training settings
    # --------------------
    # Display the loss every `display` iterations
    display = 100
    # Learning rate decay every `lr_decay` iterations
    lr_decay = 40000
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=0.1)
    # Each update use accumulated gradient by `iter_size` iterations.
    # The network only support single image input. This trick can simulate multiple image input.
    iter_size = 2
    start_iter = 1
    max_iter = cfg.TRAIN.MAX_ITER
    start_time = time.time()
    total_loss = 0
    output_dir = osp.join(cfg.DATA_DIR, "trained_model")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load checkpoint
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_iter = checkpoint["iteration"] + 1
        net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        logging.info("Loaded checkpoint from: %s" % args.checkpoint)

    # Training loop
    for iter in range(start_iter, max_iter + 1):
        optimizer.zero_grad()
        loss = 0
        for i in range(iter_size):
            data = next(data_iter)
            img = data[0].to(device)
            img_info = data[1][0].to(device)
            gt_boxes = data[2][0].to(device)
            _, _, _, _, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox, loss_oim = net(
                img, img_info, gt_boxes
            )
            loss_i = (rpn_loss_cls + rpn_loss_bbox + loss_cls + loss_bbox + loss_oim) / iter_size
            loss += loss_i.item()
            loss_i.backward()

        optimizer.step()
        scheduler.step()
        total_loss += loss

        if iter % display == 0:
            ave_loss = total_loss / display
            total_loss = 0
            logging.info(
                (
                    "\n--------------------------------------------------------------\n"
                    + "iteration: [%s / %s], loss: %.4f\n"
                    + "time cost: %.2f seconds, learning rate: %s\n"
                    + "--------------------------------------------------------------"
                )
                % (
                    iter,
                    max_iter,
                    ave_loss,
                    time.time() - start_time,
                    optimizer.param_groups[0]["lr"],
                )
            )
            start_time = time.time()

            if args.tbX:
                log_info = {
                    "loss": ave_loss,
                    "rpn_loss_cls": rpn_loss_cls,
                    "rpn_loss_bbox": rpn_loss_bbox,
                    "loss_cls": loss_cls,
                    "loss_bbox": loss_bbox,
                    "loss_oim": loss_oim,
                }
                logger.add_scalars("Train/Loss", log_info, iter)

        # Save checkpoint
        if iter % cfg.TRAIN.CHECKPOINT_ITERS == 0:
            save_name = os.path.join(output_dir, "checkpoint_iter_%s.pth" % iter)
            save_dict = {
                "iteration": iter,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(save_dict, save_name)

    if args.tbX:
        logger.close()
