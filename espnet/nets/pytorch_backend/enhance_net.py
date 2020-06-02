#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import argparse
import logging
import math
import pudb

import chainer
import numpy as np
import six
import torch
import torch.nn.functional as F

from itertools import groupby

from chainer import reporter

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.frontends.frontend import frontend_for

CTC_LOSS_THRESHOLD = 10000


def make_non_pad_mask(lengths):
    """Function to make tensor mask containing indices of the non-padded part

    e.g.: lengths = [5, 3, 2]
          mask = [[1, 1, 1, 1 ,1],
                  [1, 1, 1, 0, 0],
                  [1, 1, 0, 0, 0]]

    :param list lengths: list of lengths (B)
    :return: mask tensor containing indices of non-padded part (B, Tmax)
    :rtype: torch.Tensor
    """
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    maxlen = int(max(lengths))
    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    return seq_range_expand < seq_length_expand


class Reporter(chainer.Chain):
    """A chainer reporter wrapper"""

    def report(self, loss):
        reporter.report({'loss':loss}, self)


class BleachNet(torch.nn.Module):
    """E2E module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options
    """

    def __init__(self, idim, odim, args):
        torch.nn.Module.__init__(self)
        self.outdir = args.outdir
        self.bftype = args.bftype
        self.reporter = Reporter()
        self.frontend = frontend_for(args, idim)

        # weight initialization
        #self.init_like_chainer()
        self.loss = None
        self.loss_type = args.loss
        if self.loss_type == 'mse':
            self.loss_func = torch.nn.MSELoss()
        elif self.loss_type == 'mae':
            self.loss_func = torch.nn.L1Loss()
        elif self.loss_type == 'bce':
            self.loss_func = torch.nn.BCELoss()

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :rtype: torch.Tensor
        :rtype: float
        """
        xs_pad_ft = {'real': xs_pad['real'], 'imag': xs_pad['imag']}
        xs_pad_ft = to_torch_tensor(xs_pad_ft)
        #xs_pad_sv = {'real': xs_pad['svreal'], 'imag': xs_pad['svimag']}
        #xs_pad_sv = to_torch_tensor(xs_pad_sv)
        if 'xv' in self.bftype:
            xv=xs_pad['xv']
        else:
            xv=None
        if 'ss' in self.bftype:
            ss=xs_pad['ss']
            ilens_ss=xs_pad['ilens_ss']
        else:
            ss=None
            ilens_ss=None
        if 'angle' in self.bftype or ('ss' not in self.bftype and 'xv' not in self.bftype):
            af=xs_pad['af']
        else:
            af=None
        hs_pad, hlens, mask, _, _, _, _, _ = self.frontend(xs_pad_ft,
                                                        ilens, None,
                                                        af, xv, ss=ss,
                                                        ilens_ss=ilens_ss)
        mask_sel = to_device(self, make_non_pad_mask(ilens).unsqueeze(-1))
        ys_pad = ys_pad.masked_select(mask_sel)
        if self.loss_type == 'bce':
            mask = mask.masked_select(mask_sel)
            self.loss = self.loss_func(mask, ys_pad)
        else:
            hs_pad = hs_pad.masked_select(mask_sel)
            self.loss = self.loss_func(hs_pad, ys_pad)
        loss_data = float(self.loss)
        if not math.isnan(loss_data):
            self.reporter.report(loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def enhance(self, xs):
        """Forwarding only the frontend stage

        :param ndarray xs: input acoustic feature (T, C, F)
        """

        prev = self.training
        self.eval()
        sb=True
        xs_ft = [x['stft'] for x in xs]
        if 'xv' in self.bftype:
            xs_xv = [x['xv'] for x in xs]
            xs_xv = [to_device(self, to_torch_tensor(xx).float()) for xx in
                     xs_xv]
            xs_pad_xv = pad_list(xs_xv, 0.0)
        else:
            xs_pad_xv = None
        if 'ss' in self.bftype:
            xs_ss = [x['ss'] for x in xs]
            ilens_ss = np.fromiter((xx.shape[0] for xx in xs_ss), dtype=np.int64)
            xs_ss = [to_device(self, to_torch_tensor(xx).float()) for xx in
                     xs_ss]
            xs_pad_ss = pad_list(xs_ss, 0.0)
        else:
            xs_pad_ss = None
            ilens_ss=None
        if 'angle' in self.bftype or ('ss' not in self.bftype and 'xv' not in self.bftype):
            xs_af = [x['af'] for x in xs]
            xs_af = [to_device(self, to_torch_tensor(xx).float()) for xx in
                     xs_af]
            xs_pad_af = pad_list(xs_af, 0.0)
        else:
            xs_pad_af = None

        ilens = np.fromiter((xx.shape[0] for xx in xs_ft), dtype=np.int64)
        xs_ft = [to_device(self, to_torch_tensor(xx).float()) for xx in xs_ft]
        xs_pad_ft = pad_list(xs_ft, 0.0)
        enhanced, hlens, mask_speech, mask_noise, mask_interference, mask_post, sigma, se = self.frontend(xs_pad_ft, ilens, None, xs_pad_af, xs_pad_xv, ss=xs_pad_ss, ilens_ss=ilens_ss)
        #enhanced = enhanced.exp()
        if prev:
            self.train()
        if mask_noise is not None:
            mask_noise=mask_noise.cpu().numpy()
        if mask_post is not None:
            mask_post=mask_post.cpu().numpy()
        if sigma is not None:
            sigma=sigma.cpu().numpy()
        if mask_interference is not None:
            mask_interference=mask_interference.cpu().numpy()
        if se is not None:
            se=se.cpu().numpy()
        return enhanced.cpu().numpy(), mask_speech.cpu().numpy(), ilens, mask_noise, mask_interference, mask_post, sigma, se
