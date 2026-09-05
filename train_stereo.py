
from __future__ import print_function, division

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


import torch
import torch.nn as nn
import torch.optim as optim
from core.BANet import BANet
from evaluate_stereo import validate_sceneflow
import core.stereo_datasets as datasets
import torch.nn.functional as F


ckpt_path = './checkpoints/sceneflow'
log_path = './checkpoints/sceneflow'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_loss(disp_preds, disp_gt, valid, max_disp=192):
    """ Loss function defined over sequence of flow predictions """

    mask = (disp_gt < max_disp) & (disp_gt > 0)

    loss = 0.0
    loss += 0.3 * F.smooth_l1_loss(disp_preds[1][mask], disp_gt[mask], reduction='mean')
    loss += 1.0 * F.smooth_l1_loss(disp_preds[0][mask], disp_gt[mask], reduction='mean')

    epe = torch.abs(disp_preds[0]-disp_gt)[mask]
    epe = epe.view(-1)

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=log_path)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=log_path)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=log_path)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(BANet(args))
    print("Parameter Count: %d" % count_parameters(model))
    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")
    model.cuda()
    model.train()

    validation_frequency = 5000

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, disp_gt, valid = [x.cuda() for x in data_blob]

            assert model.training
            disp_preds = model(image1, image2, args.max_disp)
            assert model.training

            loss, metrics = compute_loss(disp_preds, disp_gt, valid, max_disp=args.max_disp)
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            logger.push(metrics)

            if total_steps % validation_frequency == validation_frequency - 1:
                save_path = Path(ckpt_path + '/%d_%s.pth' % (total_steps + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)
                # results = validate_kitti(model.module, iters=args.valid_iters)
                results = validate_sceneflow(model.module)
                logger.write_dict(results)
                model.train()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    print("FINISHED TRAINING")
    logger.close()
    PATH = ckpt_path + '/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='banet', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help="")
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0008, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 512], help="size of the random image crops used during training.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.4, 0.8], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')

    args = parser.parse_args()

    torch.manual_seed(666)
    np.random.seed(666)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path(ckpt_path).mkdir(exist_ok=True, parents=True)

    train(args)