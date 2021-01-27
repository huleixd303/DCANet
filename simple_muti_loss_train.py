import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
# from ptsemseg.augmentations import *
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

def train(cfg, writer, logger):
    # Setup seeds
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg['training'].get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug)

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']), )

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'],
                                  num_workers=cfg['training']['n_workers'],
                                  shuffle=True)

    valloader = data.DataLoader(v_loader,
                                batch_size=cfg['training']['batch_size'],
                                num_workers=cfg['training']['n_workers'])

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model = get_model(cfg['model'], n_classes).to(device)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg['training']['optimizer'].items()
                        if k != 'name'}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg['training']['resume'] is not None:
        if os.path.isfile(cfg['training']['resume']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg['training']['resume']))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True

    while i <= cfg['training']['train_iters'] and flag:
        for (images, labels) in trainloader:
            i += 1

            # Adaptive loss function
            # labels.numpy()
            # labelweights = np.zeros(2)
            # for k in range(len(labels)):
            #     tmp, _ = np.histogram(labels[k], bins=[-0.5, 0.5, 1.5])
            #     labelweights += tmp
            # labelweights = labelweights.astype(np.float32)
            # labelweights = labelweights / np.sum(labelweights)
            # # labelweights = 1 / np.log(1.1 + labelweights)
            # labelweights = 1 / labelweights
            # loss_fn.keywords['weight'] = labelweights


            start_ts = time.time()
            # scheduler.step()
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if type(outputs) == tuple:
                loss_list = []
                for k in range(len(outputs)):
                    loss_list.append(outputs[k])
                    loss_list[k] = loss_fn(input=loss_list[k], target=labels)
                if i <= 279000:
                    loss = sum(loss_list)
                    # loss_aux_0 = loss_list.pop(0)
                else:

                    # loss_aux_0 = loss_list.pop(0)

                    # loss_sum = sum(loss_list)
                    # loss_factor = [(1 - key/loss_sum)*key for key in loss_list]
                    # loss = sum(loss_factor)
                    # loss = loss_sum
                    def sigmod(x):
                        s = 1 / (1 + math.exp(-x))
                        return s
                    loss_factor = []
                    for index in range(len(loss_list)):
                        if loss_list[index] == min(loss_list):
                            loss_factor.append((1.0) * loss_list[index])
                        # elif loss_list[index] == max(loss_list):
                        #     loss_factor.append((1.0) * loss_list[index])
                        else:
                            loss_factor.append((1 - (sigmod(loss_list[index]))) * loss_list[index])
                            # loss_factor.append((1.0 - (loss_list[index])/loss_sum) * loss_list[index])
                    loss = sum(loss_factor)

                loss.backward()
                optimizer.step()
                scheduler.step()

                time_meter.update(time.time() - start_ts)

                if (i + 1) % cfg['training']['print_interval'] == 0:
                    # fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  d3_out_b2u_sig: {:.4f}  d2_out_b2u_sig: {:.4f}    d1_out_b2u_sig: {:.4f}  " \
                    #           "out_b2u_sig: {:.4f}  e3_out_t2d_sig: {:.4f}  d2_out_t2d_sig: {:.4f}  d1_out_t2d_sig: {:.4f}  " \
                    #           "out_t2d_sig: {:.4f}  finalrefuse:{:.4f}  Time/Image: {:.4f}"
                    fmt_str = "Iter [{:d}/{:d}]  loss: {:.4f}  e3_aux: {:.4f}  e2_aux: {:.4f}  e1_aux: {:.4f}  "\
                                "out: {:.4f}"
                    print_str = fmt_str.format(i + 1,
                                               cfg['training']['train_iters'],
                                               loss.item(),
                                               loss_list[0],
                                               loss_list[1],
                                               loss_list[2],
                                               loss_list[3],
                                               # loss_list[4],
                                               # loss_list[5],
                                               # loss_list[6],
                                               # loss_list[7],
                                               # loss_list[8],
                                               # loss_list[9],
                                               # loss_list[10],
                                               # loss_list[11],

                                               time_meter.avg / cfg['training']['batch_size'])

                    print(print_str)
                    logger.info(print_str)
                    writer.add_scalar('loss/train_loss', loss.item(), i + 1)
                    writer.add_scalar('loss/e3_aux', loss_list[0].item(), i + 1)
                    writer.add_scalar('loss/e2_aux', loss_list[1].item(), i + 1)
                    writer.add_scalar('loss/e1_aux', loss_list[2].item(), i + 1)
                    writer.add_scalar('loss/out', loss_list[3].item(), i + 1)

                    # writer.add_scalar('loss/train_loss', loss.item(), i + 1)
                    # writer.add_scalar('loss/d3_out_b2u_sig', loss_list[0].item(), i + 1)
                    # writer.add_scalar('loss/d2_out_b2u_sig', loss_list[1].item(), i + 1)
                    # writer.add_scalar('loss/d1_out_b2u_sig', loss_list[2].item(), i + 1)
                    # writer.add_scalar('loss/out_b2u_sig', loss_list[3].item(), i + 1)
                    # writer.add_scalar('loss/e3_out_t2d_sig', loss_list[4].item(), i + 1)
                    # writer.add_scalar('loss/d2_out_t2d_sig', loss_list[5].item(), i + 1)
                    # writer.add_scalar('loss/d1_out_t2d_sig', loss_list[6].item(), i + 1)
                    # writer.add_scalar('loss/out_t2d_sig', loss_list[7].item(), i + 1)
                    # writer.add_scalar('loss/finalrefuse', loss_list[8].item(), i + 1)


                    time_meter.reset()

            # loss_factor = []
            #
            # loss = 0.5*loss + 0.5*loss_list[-1]
            # main_branch_output = outputs[4]
            # aux_loss = loss_fn(input=aux_output, target=labels)
            # main_branch_loss = loss_fn(input=main_branch_output, target=labels)
            # loss = 0.6* aux_loss + main_branch_loss
            else:
                labels = labels.float() #for mse_loss
                loss = loss_fn(input=outputs, target=labels)

                loss.backward()
                optimizer.step()

                time_meter.update(time.time() - start_ts)

                if (i + 1) % cfg['training']['print_interval'] == 0:
                    fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                    print_str = fmt_str.format(i + 1,
                                               cfg['training']['train_iters'],
                                               loss.item(),
                                               time_meter.avg / cfg['training']['batch_size'])

                    print(print_str)
                    logger.info(print_str)
                    writer.add_scalar('loss/train_loss', loss.item(), i + 1)
                    time_meter.reset()

            # if (i + 1) % cfg['training']['val_interval'] == 0 or \
            #         (i + 1) == cfg['training']['train_iters']:
            if (((i + 1) % cfg['training']['val_interval'] == 0) and \
                    ((i + 1) >= 1)) or \
                    (i + 1) == cfg['training']['train_iters']:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs = model(images_val)
                        val_loss = loss_fn(input=outputs, target=labels_val)

                        # pred = outputs.data.max(1)[1].cpu().numpy()
                        outputs[outputs > cfg['threshold']] = 1
                        outputs[outputs <= cfg['threshold']] = 0
                        pred = outputs.data.cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar('loss/val_loss', val_loss_meter.avg, i + 1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/{}'.format(k), v, i + 1)

                for k, v in class_iou.items():
                    logger.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/cls_{}'.format(k), v, i + 1)

                val_loss_meter.reset()
                running_metrics_val.reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(writer.file_writer.get_logdir(),
                                             "{}_{}_best_model.pkl".format(
                                                 cfg['model']['arch'],
                                                 cfg['data']['dataset']))
                    torch.save(state, save_path)

            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1, 100000)

    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], str(run_id))
    print(logdir)
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')


    train(cfg, writer, logger)
