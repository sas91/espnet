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
    # loss
    parser.add_argument('--loss_type', default='msa', type=str,
                        choices=['msa'],
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
    parser.add_argument('--lr', default=1e-3, type=float,
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

    # WPE related
    parser.add_argument('--use-wpe', type=strtobool, default=False,
                        help='Apply Weighted Prediction Error')
    parser.add_argument('--use-wpe-test', type=strtobool, default=False,
                        help='Apply Weighted Prediction Error')
    parser.add_argument('--wtype', default='blstmp', type=str,
                        choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm',
                                 'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'],
                        help='Type of encoder network architecture '
                             'of the mask estimator for WPE. '
                             '')
    parser.add_argument('--wlayers', type=int, default=2,
                        help='')
    parser.add_argument('--wunits', type=int, default=300,
                        help='')
    parser.add_argument('--wprojs', type=int, default=300,
                        help='')
    parser.add_argument('--wdropout-rate', type=float, default=0.0,
                        help='')
    parser.add_argument('--wpe-taps', type=int, default=5,
                        help='')
    parser.add_argument('--wpe-delay', type=int, default=3,
                        help='')
    parser.add_argument('--use-dnn-mask-for-wpe', type=strtobool,
                        default=False,
                        help='Use DNN to estimate the power spectrogram. '
                             'This option is experimental.')

    # Beamformer related
    parser.add_argument('--use-beamformer', type=strtobool,
                        default=True, help='')
    parser.add_argument('--btype', default='blstmp', type=str,
                        choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm',
                                 'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'],
                        help='Type of encoder network architecture '
                             'of the mask estimator for Beamformer.')
    parser.add_argument('--bftype', default='mvdr', type=str,
                        choices=['mvdr', 'sb_mvdr', 'sb_mvdr_pm', 'mvdr_xv',
                                 'mask', 'mask_xv', 'mag_reg', 'mag_reg_xv',
                                 'mask_sb_ss','mask_ss_attention_2',
                                 'mask_ss', 'mask_ss_attention', 'mask_angle_ss',
                                 'mvdr_xv_pm', 'mvdr_xv_sb', 'gdr', 'gdr_pm', 'lcmv',
                                 'sb_mvdr_pm_ad', 'sb_mvdr_ad', 'gdr_ad', 'gdr_pm_ad'],
                        help='Beamforming algorithm')
    parser.add_argument('--loss', default='mse', type=str,
                        choices=['mse', 'mae', 'bce', 'mse_log'],
                        help='Loss function')
    parser.add_argument('--blayers', type=int, default=2,
                        help='')
    parser.add_argument('--bunits', type=int, default=300,
                        help='')
    parser.add_argument('--bprojs', type=int, default=300,
                        help='')
    parser.add_argument('--badim', type=int, default=320,
                        help='')
    parser.add_argument('--ref-channel', type=int, default=-1,
                        help='The reference channel used for beamformer. '
                             'By default, the channel is estimated by DNN.')
    parser.add_argument('--bdropout-rate', type=float, default=0.0,
                        help='')

    args, _ = parser.parse_known_args(cmd_args)

    from espnet.utils.dynamic_import import dynamic_import
    if args.model_module is not None:
        model_class = dynamic_import(args.model_module)
        model_class.add_arguments(parser)
    args = parser.parse_args(cmd_args)
    if args.model_module is None:
        args.model_module = "espnet.nets." + args.backend + "_backend.enhance_net:BleachNet"
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
            from espnet.enhance.pytorch_backend.enhance import train
            train(args)
        else:
            raise ValueError("Only pytorch is supported.")


if __name__ == '__main__':
    main(sys.argv[1:])
