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
from espnet.nets.pytorch_backend.e2e_asr_mix import PIT
from espnet.nets.pytorch_backend.frontends.mask_estimator import MaskEstimator

from espnet.nets.pytorch_backend.frontends.beamformer \
    import apply_beamforming_vector
from espnet.nets.pytorch_backend.frontends.beamformer \
    import get_mvdr_vector
from espnet.nets.pytorch_backend.frontends.beamformer \
    import get_gev_vector
from espnet.nets.pytorch_backend.frontends.beamformer \
    import blind_analytic_normalization
from espnet.nets.pytorch_backend.frontends.beamformer \
    import get_power_spectral_density_matrix
from torch_complex.tensor import ComplexTensor

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

    def __init__(self, idim, args):
        torch.nn.Module.__init__(self)
        self.outdir = args.outdir
        self.bftype = args.bftype
        self.btype = args.btype
        self.bidim = idim
        self.blayers = args.blayers
        self.bunits = args.bunits
        self.bprojs = args.bprojs
        if 'pit' in self.bftype:
            self.num_spkrs = 4
            self.pit = PIT(self.num_spkrs)
            self.mask = MaskEstimator(self.btype, self.bidim,
                                      self.blayers, self.bunits, self.bprojs,
                                      0.0, nmask=5, only_mask=True)
        elif 'denoise' in self.bftype:
            self.num_spkrs = 1
            self.mask = MaskEstimator(self.btype, self.bidim,
                                      self.blayers, self.bunits, self.bprojs,
                                      0.0, nmask=2, only_mask=True)
        elif 'ts' in self.bftype:
            self.num_spkrs = 1
            self.mask = MaskEstimator(self.btype, self.bidim,
                                      self.blayers, self.bunits, self.bprojs,
                                      0.0, nmask=2, SA=True, SA_type='lhuc_ss', only_mask=True)
        elif 'xvec' in self.bftype:
            if 'lhuc' in self.bftype:
                self.num_spkrs = 1
                self.mask = MaskEstimator(self.btype, self.bidim,
                                          self.blayers, self.bunits, self.bprojs,
                                          0.0, nmask=2, SA=True, SA_type='lhuc', only_mask=True)
            else:
                self.num_spkrs = 1
                self.mask = MaskEstimator(self.btype, self.bidim,
                                          self.blayers, self.bunits, self.bprojs,
                                          0.0, nmask=2, SA=True, SA_type='cat', only_mask=True)
        self.reporter = Reporter()

        # weight initialization
        #self.init_like_chainer()
        self.loss = None
        self.loss_type = args.loss_type
        if self.loss_type == 'mse':
            self.loss_func = torch.nn.MSELoss()
        elif self.loss_type == 'mae':
            self.loss_func = torch.nn.L1Loss()
        elif self.loss_type == 'bce':
            self.loss_func = torch.nn.BCELoss()

    def forward(self, xs_pad_ft, ilens, ys_pad):
        """E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :rtype: torch.Tensor
        :rtype: float
        """
        mask_sel = to_device(self, make_non_pad_mask(ilens).unsqueeze(-1))
        if 'pit' in self.bftype:
            ys = [None] * self.num_spkrs
            hs = [None] * self.num_spkrs
            ys[0] = ys_pad['s1'].masked_select(mask_sel)
            ys[1] = ys_pad['s2'].masked_select(mask_sel)
            ys[2] = ys_pad['s3'].masked_select(mask_sel)
            ys[3] = ys_pad['s4'].masked_select(mask_sel)
            yn=ys_pad['n'].masked_select(mask_sel)
            if self.loss_type == 'bce':
                (hs[0], hs[1], hs[2], hs[3], hn), _, _ = self.mask(xs_pad_ft, ilens)
                hs[0] = hs[0].masked_select(mask_sel)
                hs[1] = hs[1].masked_select(mask_sel)
                hs[2] = hs[2].masked_select(mask_sel)
                hs[3] = hs[3].masked_select(mask_sel)
                hn = hn.masked_select(mask_sel)
            else:
                ms = [None] * self.num_spkrs
                (ms[0], ms[1], ms[2], ms[3], mn), _, _ = self.mask(xs_pad_ft, ilens)
                hs[0] = (xs_pad_ft * ms[0]).masked_select(mask_sel)
                hs[1] = (xs_pad_ft * ms[1]).masked_select(mask_sel)
                hs[2] = (xs_pad_ft * ms[2]).masked_select(mask_sel)
                hs[3] = (xs_pad_ft * ms[3]).masked_select(mask_sel)
                hn = (xs_pad_ft * mn).masked_select(mask_sel)
            loss_perm = torch.stack([self.loss_func(hs[i // self.num_spkrs],
                                                    ys[i % self.num_spkrs])
                                     for i in range(self.num_spkrs ** 2)], dim=0)
            loss_pit, min_perm = self.pit.min_pit_sample(loss_perm)
            self.loss = 0.8 * loss_pit + 0.2 * self.loss_func(hn, yn)
        elif 'ts' in self.bftype:
            ss=ys_pad['ss']
            ilens_ss=ys_pad['ilens_ss']
            ys = ys_pad['s1'].masked_select(mask_sel)
            yn = ys_pad['n'].masked_select(mask_sel)
            (ms, mn), _, _ = self.mask(xs_pad_ft, ilens, ss=ss, ilens_ss=ilens_ss)
            hs = (xs_pad_ft * ms).masked_select(mask_sel)
            hn = (xs_pad_ft * mn).masked_select(mask_sel)
            self.loss = 0.9 * self.loss_func(hs, ys) + 0.1 * self.loss_func(hn, yn)
        elif 'xvec' in self.bftype:
            xv=ys_pad['xv']
            ys = ys_pad['s1'].masked_select(mask_sel)
            yn = ys_pad['n'].masked_select(mask_sel)
            (ms, mn), _, _ = self.mask(xs_pad_ft, ilens, xv=xv)
            hs = (xs_pad_ft * ms).masked_select(mask_sel)
            hn = (xs_pad_ft * mn).masked_select(mask_sel)
            self.loss = 0.9 * self.loss_func(hs, ys) + 0.1 * self.loss_func(hn, yn)
        else:
            ys = ys_pad['s1'].masked_select(mask_sel)
            yn = ys_pad['n'].masked_select(mask_sel)
            (ms, mn), _, _ = self.mask(xs_pad_ft, ilens)
            hs = (xs_pad_ft * ms).masked_select(mask_sel)
            hn = (xs_pad_ft * mn).masked_select(mask_sel)
            self.loss = 0.5 * self.loss_func(hs, ys) + 0.5 * self.loss_func(hn, yn)
        loss_data = float(self.loss)
        if not math.isnan(loss_data):
            self.reporter.report(loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def enhance(self, xs, use_bf_test=False):
        """Forwarding only the frontend stage

        :param ndarray xs: input acoustic feature (T, C, F)
        """

        prev = self.training
        self.eval()
        if use_bf_test:
            xs_ft = xs
        else:
            xs_ft = [np.abs(x['stft']) for x in xs]
        xs_ft = [to_device(self, to_torch_tensor(xx).float()) for xx in xs_ft]
        xs_pad_ft = pad_list(xs_ft, 0.0)
        if 'xvec' in self.bftype:
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
        if use_bf_test:
            xs_pad_ft = xs_pad_ft.permute(0, 3, 2, 1)
        ilens = np.fromiter((xx.shape[0] for xx in xs_ft), dtype=np.int64)
        ilens = torch.from_numpy(np.asarray(ilens)).to(xs_pad_ft.device)
        ms2=None
        ms3=None
        ms4=None
        enhanced=None
        if 'pit' in self.bftype:
            (ms1, ms2, ms3, ms4, mn), _, _ = self.mask(xs_pad_ft, ilens, use_bf_test=True)
        elif 'xvec' in self.bftype:
            (ms1, mn), _, _ = self.mask(xs_pad_ft, ilens, xv=xs_pad_xv, use_bf_test=False)
        elif 'ss' in self.bftype:
            (ms1, mn), _, _ = self.mask(xs_pad_ft, ilens, ss=xs_pad_ss, ilens_ss=ilens_ss, use_bf_test=False)
        else:
            (ms1, mn), _, _ = self.mask(xs_pad_ft, ilens, use_bf_test=True)
            if use_bf_test:
                psd_speech = get_power_spectral_density_matrix(xs_pad_ft, ms1)
                psd_noise = get_power_spectral_density_matrix(xs_pad_ft, mn)
                u = torch.zeros(*(xs_pad_ft.size()[:-3] + (xs_pad_ft.size(-2),)), device=xs_pad_ft.device)
                u[..., 0].fill_(1)

                ws = get_mvdr_vector(psd_speech, psd_noise, u)
                #ws_gev = get_gev_vector(psd_speech, psd_noise)
                #ws_gev = blind_analytic_normalization(ws_gev, psd_noise)
                enhanced = apply_beamforming_vector(ws, xs_pad_ft)
                enhanced = enhanced.transpose(-1, -2)
                #enhanced_gev = apply_beamforming_vector(ws_gev, xs_pad_ft)
                #enhanced_gev = enhanced_gev.transpose(-1, -2)
                ms1 = ms1.transpose(-1, -2)
                mn = mn.transpose(-1, -2)
        if prev:
            self.train()
        if ms2 is not None:
            ms2=ms2.cpu().numpy()
        if ms3 is not None:
            ms3=ms3.cpu().numpy()
        if ms4 is not None:
            ms4=ms4.cpu().numpy()
        if enhanced is not None:
            enhanced=enhanced.cpu().numpy()
        return ms1.cpu().numpy(), mn.cpu().numpy(), ms2, ms3, ms4, enhanced, ilens.cpu().numpy()
