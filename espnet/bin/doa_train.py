#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import logging
import os
import platform
import random
import subprocess
import sys

import numpy as np

from espnet.utils.cli_utils import strtobool


def main(cmd_args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general configuration
    parser.add_argument('--ngpu', default=0, type=int,
                        help='Number of GPUs')
    parser.add_argument('--backend', default='pytorch', type=str,
                        choices=['pytorch'],
                        help='Backend library')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--debugdir', type=str,
                        help='Output directory for debugging')
    parser.add_argument('--resume', '-r', default='', nargs='?',
                        help='Resume the training from snapshot')
    parser.add_argument('--minibatches', '-N', type=int, default='-1',
                        help='Process only N minibatches (for debug)')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--tensorboard-dir', default=None, type=str, nargs='?', help="Tensorboard log dir path")
    # task related
    parser.add_argument('--train-json', type=str, default=None,
                        help='Filename of train label data (json)')
    parser.add_argument('--valid-json', type=str, default=None,
                        help='Filename of validation label data (json)')
    # network architecture
    parser.add_argument('--model-module', type=str, default=None,
                        help='model defined module (default: espnet.nets.xxx_backend.e2e_asr:E2E)')
    parser.add_argument('--num-spkrs', default=1, type=int,
                        choices=[1, 2],
                        help='Number of speakers in the speech.')
    parser.add_argument('--resolution', default=1, type=int,
                        choices=[1, 2, 3, 4, 5, 6, 10],
                        help='Angle resolution.')
    parser.add_argument('--arch-type', default=1, type=int,
                        choices=[1, 2, 3, 4],
                        help='CNN Architecture Type.')
    # loss
    parser.add_argument('--loss-type', default='sce', type=str,
                        choices=['ce', 'emd', 'sce', 'bce'],
                        help='Type of loss.')
    # minibatch related
    parser.add_argument('--sortagrad', default=0, type=int, nargs='?',
                        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs")
    parser.add_argument('--batch-size', '-b', default=50, type=int,
                        help='Batch size')
    parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--n-iter-processes', default=0, type=int,
                        help='Number of processes of iterator')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    # optimization related
    parser.add_argument('--opt', default='adam', type=str,
                        choices=['adadelta', 'adam', 'noam'],
                        help='Optimizer')
    parser.add_argument('--accum-grad', default=1, type=int,
                        help='Number of gradient accumuration')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='Learning rate for optimizer')
    parser.add_argument('--eps', default=1e-6, type=float,
                        help='Epsilon constant for optimizer')
    parser.add_argument('--weight-decay', default=0.0, type=float,
                        help='Weight decay ratio')
    parser.add_argument('--criterion', default='acc', type=str,
                        choices=['loss', 'acc'],
                        help='Criterion to perform epsilon decay')
    parser.add_argument('--threshold', default=1e-4, type=float,
                        help='Threshold to stop iteration')
    parser.add_argument('--epochs', '-e', default=30, type=int,
                        help='Maximum number of epochs')
    parser.add_argument('--early-stop-criterion', default='validation/main/acc', type=str, nargs='?',
                        help="Value to monitor to trigger an early stopping of the training")
    parser.add_argument('--patience', default=3, type=int, nargs='?',
                        help="Number of epochs to wait without improvement before stopping the training")
    parser.add_argument('--grad-clip', default=5, type=float,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--dropout-rate', default=0.0, type=float,
                        help='Dropout rate for the encoder')
    parser.add_argument('--llayers', type=int, default=2,
                        help='')
    parser.add_argument('--lunits', type=int, default=300,
                        help='')
    parser.add_argument('--lprojs', type=int, default=300,
                        help='')

    args, _ = parser.parse_known_args(cmd_args)

    from espnet.utils.dynamic_import import dynamic_import
    if args.model_module is not None:
        model_class = dynamic_import(args.model_module)
        model_class.add_arguments(parser)
    args = parser.parse_args(cmd_args)
    if args.model_module is None:
        args.model_module = "espnet.nets." + args.backend + "_backend.doa_net:LocalizationNet"
    if 'pytorch_backend' in args.model_module:
        args.backend = 'pytorch'

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = subprocess.check_output(["free-gpu", "-n", str(args.ngpu)]).decode().strip()
        logging.info('ID: use gpu' + cvd)
        os.environ['CUDA_VISIBLE_DEVICES'] = cvd
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # set random seed
    logging.info('random seed = %d' % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # train
    logging.info('backend = ' + args.backend)
    if args.num_spkrs == 1:
        if args.backend == "pytorch":
            from espnet.doa.pytorch_backend.doa import train
            train(args)
        else:
            raise ValueError("Only pytorch is supported.")


if __name__ == '__main__':
    main(sys.argv[1:])
