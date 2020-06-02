#!/usr/bin/env python
import argparse
from distutils.util import strtobool
import logging
import os
import random
import sys

import numpy as np

from espnet.doa.pytorch_backend.doa import classify
from espnet.utils.cli_utils import strtobool


def main(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general configuration
    parser.add_argument('--ngpu', default=0, type=int,
                        help='Number of GPUs')
    parser.add_argument('--backend', default='chainer', type=str,
                        choices=['chainer', 'pytorch'],
                        help='Backend library')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--verbose', '-V', default=1, type=int,
                        help='Verbose option')
    parser.add_argument('--batchsize', default=1, type=int,
                        help='Batch size for beam search (0: means no batch processing)')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    # task related
    parser.add_argument('--recog-json', type=str,
                        help='Filename of recognition data (json)')
    parser.add_argument('--resolution', default=1, type=int,
                        choices=[1, 2, 3, 4, 5, 6, 10],
                        help='Angle resolution.')
    parser.add_argument('--resolution-test', default=6, type=int,
                        choices=[1, 2, 3, 4, 5, 6, 8, 10],
                        help='Angle resolution.')
    # model (parameter) related
    parser.add_argument('--model', type=str, required=True,
                        help='Model file parameters to read')
    parser.add_argument('--model-conf', type=str, default=None,
                        help='Model config file')

    # Outputs configuration
    parser.add_argument('--doa-dir', type=str, default=None,
                        help='The directory saving the DOA.')

    args = parser.parse_args(args)

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

        # TODO(kamo): support of multiple GPUs
        if args.ngpu > 1:
            logging.error("The program only supports ngpu=1.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info('set random seed = %d' % args.seed)

    # recog
    logging.info('backend = ' + args.backend)
    if args.backend == "pytorch":
        classify(args)
    else:
        raise ValueError("Only pytorch is supported.")


if __name__ == '__main__':
    main(sys.argv[1:])
