import os
import sys
sys.path.append('/media/hulei/disk/hulei/Coding/pytorch-semseg')

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
import nni
from nni.utils import merge_parameter
from nni.utils import merge_parameter
import logging
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
logger = logging.getLogger('WHU_AutoML')

def main(cfg):
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
    if cfg['training']['optimizer']['name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['sgd_momentum'],
                                    weight_decay=cfg['sgd_weight_decay'])
    # optimizer_cls = get_optimizer(cfg)
    # optimizer_params = {k: v for k, v in cfg['training']['optimizer'].items()
    #                     if k != 'name'}

    # optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    val_loss_meter = averageMeter()
    best_iou = -100.0

    i = start_iter
    flag = True

    while i <= cfg['training']['train_iters'] and flag:
        for (images, labels) in trainloader:
            i += 1

            scheduler.step()
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # loss = loss_fn(input=outputs, target=labels)
            if  cfg['model']['arch'] == "SARunet":
                labels = labels.float()
                labels = labels.unsqueeze(0)
                labels_aug = F.interpolate(labels, scale_factor=0.5, mode='nearest').long().squeeze(0)
                labels = labels.long().squeeze(0)
                labels_aug = labels_aug.repeat(1, 2, 2)
                loss_main = loss_fn(outputs[0], labels)
                loss_aug = loss_fn(outputs[1], labels_aug)
                loss = loss_main + loss_aug

            else:

                loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            # if (i + 1) % cfg['training']['val_interval'] == 0 or \
            #         (i + 1) == cfg['training']['train_iters']:

            if ((i + 1) % cfg['training']['val_interval'] == 0):
                print('Train Iter: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i+1, i+1, cfg['training']['train_iters'],
                           100. * (i+1) / cfg['training']['train_iters'], loss.item()))
                logger.info('Train Iter: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i+1, i+1, cfg['training']['train_iters'],
                           100. * (i+1) / cfg['training']['train_iters'], loss.item()))

            if (((i + 1) % cfg['training']['val_interval'] == 0) and \
                    ((i + 1) >= 1)) or \
                    (i + 1) == cfg['training']['train_iters']:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs = model(images_val)
                        if cfg['model']['arch'] == "SARunet":
                            outputs = outputs[0]
                            val_loss = loss_fn(input=outputs, target=labels_val)

                        else:

                            val_loss = loss_fn(input=outputs, target=labels_val)

                        # val_loss = loss_fn(input=outputs, target=labels_val)
                        # pred = outputs.data.max(1)[1].cpu().numpy()


                        outputs[outputs > cfg['threshold']] = 1
                        outputs[outputs <= cfg['threshold']] = 0
                        pred = outputs.data.cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                # writer.add_scalar('loss/val_loss', val_loss_meter.avg, i + 1)
                print("Iter %d ValLoss: %.4f" % (i + 1, val_loss_meter.avg))
                logger.info("Iter %d ValLoss: %.4f" % (i + 1, val_loss_meter.avg))

                _, class_iou = running_metrics_val.get_scores()
                IoU = class_iou[1]

                val_loss_meter.reset()
                running_metrics_val.reset()

                if IoU > best_iou:
                    best_iou = IoU

                nni.report_intermediate_result(IoU)
                logger.info('test IoU %g', IoU)
                logger.debug('Pipe send intermediate result done.')


            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break
    nni.report_final_result(best_iou)
    logger.debug('Final result is %g', best_iou)
    logger.debug('Send final result done.')

def get_params():
    # parser = argparse.ArgumentParser(description="config")
    # parser.add_argument(
    #     "--config_hp",
    #     nargs="?",
    #     type=str,
    #
    #     help="Configuration file to use"
    # )
    #
    # args = parser.parse_args()



    with open('Road_HPO_dif.yml') as fp:
        cfg = yaml.load(fp)

    return cfg



if __name__ == "__main__":
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()


        # tuner_params['training']['lr_schedule']['max_iter'] = tuner_params['TRIAL_BUDGET'] * 296

        params = (get_params())
        params['training']['train_iters'] = tuner_params['TRIAL_BUDGET'] * 296
        # a= {'lc': 5}
        # params.update(a)
        params.update(tuner_params)
        # params = merge_parameter(get_params(), tuner_params)
        # logger.info(params)
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise


