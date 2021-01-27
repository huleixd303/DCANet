"""
Example 5 - MNIST
=================

Small CNN for MNIST implementet in both Keras and PyTorch.
This example also shows how to log results to disk during the optimization
which is useful for long runs, because intermediate results are directly available
for analysis. It also contains a more realistic search space with different types
of variables to be optimized.

"""
import os
import pickle
import argparse
import sys
sys.path.append('/media/hulei/disk/hulei/Coding/HpBandSter')

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB
from torch.utils.tensorboard import SummaryWriter
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
import yaml

import shutil

import random

import logging
logging.basicConfig(level=logging.DEBUG)



# parser = argparse.ArgumentParser(description='train on dataset')
# parser.add_argument('--min_budget',   type=float, help='Minimum number of epochs for training.',    default=1)
# parser.add_argument('--max_budget',   type=float, help='Maximum number of epochs for training.',    default=9)
# parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=16)
# parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true', default=False)
# parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
# parser.add_argument('--nic_name',type=str, help='Which network interface to use for communication.', default='lo')
# parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default='.')
# parser.add_argument('--backend',help='Toggles which worker is used. Choose between a pytorch and a keras implementation.', choices=['pytorch', 'keras'], default='pytorch')

# args=parser.parse_args()




parser = argparse.ArgumentParser(description="config")
parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Configuration file to use"
    )

args=parser.parse_args()


	
with open(args.config) as fp:
	cfg = yaml.load(fp)

if cfg['backend'] == 'pytorch':
	from HPO_Worker import PyTorchWorker as worker
else:
	from hpbandster.examples.example_5_keras_worker import KerasWorker as worker

# run_id = random.randint(1, 100000)
# logdir = os.path.join('runs', os.path.basename(args.config)[:-4], str(run_id))
# # writer = SummaryWriter(log_dir=logdir)
#
# print('RUNDIR: {}'.format(logdir))
# shutil.copy(args.config, logdir)
#
# logger = get_logger(logdir)
# logger.info('Let the games begin')


# Every process has to lookup the hostname
host = hpns.nic_name_to_host(cfg['nic_name'])


if cfg['worker']:
	import time
	time.sleep(5)	# short artificial delay to make sure the nameserver is already running
	# w = worker(cfg=cfg, run_id=cfg['run_id'], host=host, timeout=120, writer=writer)
	w = worker(cfg=cfg, run_id=cfg['run_id'], host=host, timeout=120)
	w.load_nameserver_credentials(working_directory=cfg['shared_directory'])
	w.run(background=False)
	exit(0)


# This example shows how to log live results. This is most useful
# for really long runs, where intermediate results could already be
# interesting. The core.result submodule contains the functionality to
# read the two generated files (results.json and configs.json) and
# create a Result object.
result_logger = hpres.json_result_logger(directory=cfg['shared_directory'], overwrite=False)


# Start a nameserver:
NS = hpns.NameServer(run_id=cfg['run_id'], host=host, port=0, working_directory=cfg['shared_directory'])
ns_host, ns_port = NS.start()

# Start local worker
# w = worker(run_id=cfg['run_id'], host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=120, logger=logger, cfg=cfg, writer=writer)
w = worker(run_id=cfg['run_id'], host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=120, cfg=cfg)
w.run(background=True)

# Run an optimizer
bohb = BOHB(  configspace = worker.get_configspace(),
			  run_id = cfg['run_id'],
			  host=host,
			  nameserver=ns_host,
			  nameserver_port=ns_port,
			  result_logger=result_logger,
			  min_budget=cfg['min_budget'], max_budget=cfg['max_budget'],
		   )
res = bohb.run(n_iterations=cfg['n_iterations'])


# store results
with open(os.path.join(cfg['shared_directory'], 'results.pkl'), 'wb') as fh:
	pickle.dump(res, fh)




# shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

