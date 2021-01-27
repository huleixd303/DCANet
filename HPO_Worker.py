import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
# sys.path.append('/media/hulei/disk/hulei/Coding/HpBandSter')
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

from torch.utils.tensorboard import SummaryWriter

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)
#
# parser = argparse.ArgumentParser(description="config")
# parser.add_argument(
#         "--config",
#         nargs="?",
#         type=str,
#         default="configs/fcn8s_pascal.yml",
#         help="Configuration file to use"
#     )
#
# args = parser.parse_args()
#
# with open(args.config) as fp:
#     cfg = yaml.load(fp)
#
# run_id = random.randint(1, 100000)
# logdir = os.path.join('runs', os.path.basename(args.config)[:-4], str(run_id))
# writer = SummaryWriter(log_dir=logdir)
#
# print('RUNDIR: {}'.format(logdir))
# shutil.copy(args.config, logdir)
#
# logger = get_logger(logdir)
# logger.info('Let the games begin')

class PyTorchWorker(Worker):
    def __init__(self, cfg,  **kwargs):
        super().__init__(**kwargs)

        # self.writer = writer
        self.cfg = cfg
        # self.logger = logger

        # Setup Augmentations
        augmentations = self.cfg['training'].get('augmentations', None)
        data_aug = get_composed_augmentations(augmentations)

        # Setup Dataloader
        data_loader = get_loader(self.cfg['data']['dataset'])
        data_path = self.cfg['data']['path']

        t_loader = data_loader(
            data_path,
            is_transform=True,
            split=self.cfg['data']['train_split'],
            img_size=(self.cfg['data']['img_rows'], cfg['data']['img_cols']),
            augmentations=data_aug)

        v_loader = data_loader(
            data_path,
            is_transform=True,
            split=self.cfg['data']['val_split'],
            img_size=(self.cfg['data']['img_rows'], cfg['data']['img_cols']), )

        self.n_classes = t_loader.n_classes
        self.trainloader = data.DataLoader(t_loader,
                                      batch_size=self.cfg['training']['batch_size'],
                                      num_workers=self.cfg['training']['n_workers'],
                                      shuffle=True)

        self.valloader = data.DataLoader(v_loader,
                                    batch_size=self.cfg['training']['batch_size'],
                                    num_workers=self.cfg['training']['n_workers'])

    def compute(self, config, budget, working_directory, *args, **kwargs):
        # Setup seeds
        torch.manual_seed(self.cfg.get('seed', 1337))
        torch.cuda.manual_seed(self.cfg.get('seed', 1337))
        np.random.seed(self.cfg.get('seed', 1337))
        random.seed(self.cfg.get('seed', 1337))

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup Model
        model = get_model(self.cfg['model'], self.n_classes).to(device)

        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

        # Setup Metrics
        running_metrics_val = runningScore(self.n_classes)

        # Setup optimizer, lr_scheduler and loss function
        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'], weight_decay=config['sgd_weight_decay'])
        self.logger.info("Using optimizer {}".format(optimizer))

        scheduler = get_scheduler(optimizer, self.cfg['training']['lr_schedule'])

        loss_fn = get_loss_function(self.cfg)
        # self.logger.info("Using loss {}".format(loss_fn))

        start_iter = 0
        val_loss_meter = averageMeter()
        time_meter = averageMeter()

        # best_iou = -100.0
        i = start_iter
        flag = True

        while i <= (budget*296) and flag:
            for (images, labels) in self.trainloader:
                i += 1
                start_ts = time.time()
                scheduler.step()
                model.train()
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                loss = loss_fn(input=outputs, target=labels)

                loss.backward()
                optimizer.step()

                time_meter.update(time.time() - start_ts)

                # if (i + 1) == self.cfg['training']['train_iters']:
                #     flag = False
                #     break

                # if (i + 1) % self.cfg['training']['print_interval'] == 0:
                #     fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                #     print_str = fmt_str.format(i + 1,
                #                                self.cfg['training']['train_iters'],
                #                                loss.item(),
                #                                time_meter.avg / self.cfg['training']['batch_size'])
                #
                #     print(print_str)
                #     self.logger.info(print_str)
                #     self.writer.add_scalar('loss/train_loss', loss.item(), i + 1)
                #     time_meter.reset()

                # if (i + 1) % cfg['training']['val_interval'] == 0 or \
                #         (i + 1) == cfg['training']['train_iters']:
                # if (((i + 1) % self.cfg['training']['val_interval'] == 0) and \
                #     ((i + 1) >= 1)) or \
                #         (i + 1) == (budget*296):
        model.eval()
        with torch.no_grad():
            for i_val, (images_val, labels_val) in tqdm(enumerate(self.valloader)):
                images_val = images_val.to(device)
                labels_val = labels_val.to(device)

                outputs = model(images_val)
                val_loss = loss_fn(input=outputs, target=labels_val)

                # pred = outputs.data.max(1)[1].cpu().numpy()
                outputs[outputs > config['threshold']] = 1
                outputs[outputs <= config['threshold']] = 0
                pred = outputs.data.cpu().numpy()
                gt = labels_val.data.cpu().numpy()

                running_metrics_val.update(gt, pred)
                val_loss_meter.update(val_loss.item())
        val_loss_meter = val_loss_meter.avg
        # self.writer.add_scalar('loss/val_loss', val_loss_meter.avg, i + 1)
        # self.logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

        score, class_iou = running_metrics_val.get_scores()
        # for k, v in score.items():
        #     print(k, v)
        #     self.logger.info('{}: {}'.format(k, v))
        #     self.writer.add_scalar('val_metrics/{}'.format(k), v, i + 1)
        #
        # for k, v in class_iou.items():
        #     self.logger.info('{}: {}'.format(k, v))
        #     self.writer.add_scalar('val_metrics/cls_{}'.format(k), v, i + 1)
        # result = ({

        # return ({
        #     'loss': 1 - val_loss_meter,  # remember: HpBandSter always minimizes!
        #     'info': {'IoU': class_iou,
        #              'Overall Acc': score["Overall Acc: \t"],
        #              'Mean Acc': score["Mean Acc: \t"],
        #              'FreqW Acc': score["FreqW Acc: \t"],
        #              'Mean IoU': score["Mean IoU: \t"],
        #              }
        #
        # })

        result = ({
            'loss': val_loss_meter,  # remember: HpBandSter always minimizes!
            'IoU': class_iou,
            'info': score

        })

        val_loss_meter.reset()
        running_metrics_val.reset()
        return result

        # if score["Mean IoU : \t"] >= best_iou:
        #     best_iou = score["Mean IoU : \t"]
        #     state = {
        #         "epoch": i + 1,
        #         "model_state": model.state_dict(),
        #         "optimizer_state": optimizer.state_dict(),
        #         "scheduler_state": scheduler.state_dict(),
        #         "best_iou": best_iou,
        #     }
        #     save_path = os.path.join(self.writer.file_writer.get_logdir(),
        #                              "{}_{}_best_model.pkl".format(
        #                                  self.cfg['model']['arch'],
        #                                  self.cfg['data']['dataset']))
        #     torch.save(state, save_path)



    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter('lr', lower=2e-3, upper=6e-3, default_value='4e-3', log=True)
        threshold = CSH.UniformFloatHyperparameter('threshold', lower=0.5, upper=0.7, default_value='0.5', log=True)

        # For demonstration purposes, we add different optimizers as categorical hyperparameters.
        # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
        # SGD has a different parameter 'momentum'.
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.85, upper=0.99, default_value=0.9,
                                                      log=False)
        sgd_weight_decay = CSH.UniformFloatHyperparameter('sgd_weight_decay', lower=0.0005, upper=0.01, default_value=0.002,
                                                      log=False)

        cs.add_hyperparameters([lr, threshold, optimizer, sgd_momentum, sgd_weight_decay])

        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        cond_momentum = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond_momentum)
        cond_weight_decay = CS.EqualsCondition(sgd_weight_decay, optimizer, 'SGD')
        cs.add_condition(cond_weight_decay)

        return cs


# if __name__ == "__main__":
#
#     worker = PyTorchWorker(run_id='0', cfg=cfg)
#     cs = worker.get_configspace()
#
#     config_HPO = cs.sample_configuration().get_dictionary()
#     print(config_HPO)
#     res = worker.compute(config=config_HPO, budget=cfg['training']['train_iters'], working_directory='.')
#     # print(res)

    # train(cfg, writer, logger)