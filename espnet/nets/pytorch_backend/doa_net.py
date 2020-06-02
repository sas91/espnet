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
from espnet.nets.pytorch_backend.rnn.encoders import RNNP
from espnet.nets.pytorch_backend.rnn.encoders import VGG2L
from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.e2e_asr_mix import PIT
from espnet.nets.pytorch_backend.frontends.mask_estimator import MaskEstimator


CTC_LOSS_THRESHOLD = 10000


class SCE(torch.nn.Module):
    def __init__(self, classes=360):
        super(SCE, self).__init__()
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=1)
        true_dist = torch.zeros_like(pred)
        if self.classes == 360:
            true_dist.scatter_(1, target.data.unsqueeze(1), 0.4)
            true_dist.scatter_(1, (target.data.unsqueeze(1)-1)%self.classes, 0.2)
            true_dist.scatter_(1, (target.data.unsqueeze(1)+1)%self.classes, 0.2)
            true_dist.scatter_(1, (target.data.unsqueeze(1)-2)%self.classes, 0.1)
            true_dist.scatter_(1, (target.data.unsqueeze(1)+2)%self.classes, 0.1)
        else:
            true_dist.scatter_(1, target.data.unsqueeze(1), 0.6)
            true_dist.scatter_(1, (target.data.unsqueeze(1)-1)%self.classes, 0.2)
            true_dist.scatter_(1, (target.data.unsqueeze(1)+1)%self.classes, 0.2)
        return torch.mean(torch.sum(-true_dist * pred, dim=1))


class EMD(torch.nn.Module):
    def __init__(self, classes=360, norm=2, soft=False):
        super(EMD, self).__init__()
        self.classes = classes
        self.norm = norm
        self.soft = soft

    def forward(self, predictions, targets):
        predictions_p = F.softmax(predictions, dim=1)
        predictions_c = torch.cumsum(predictions_p, dim=1)
        true_dist = torch.zeros_like(predictions_c)
        if self.soft:
            true_dist.scatter_(1, targets.data.unsqueeze(1), 0.6)
            true_dist.scatter_(1, (targets.data.unsqueeze(1)-1)%self.classes, 0.2)
            true_dist.scatter_(1, (targets.data.unsqueeze(1)+1)%self.classes, 0.2)
        else:
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0)
        true_dist = torch.cumsum(true_dist, dim=1)
        lossvalue = torch.norm(predictions_c - true_dist, p=self.norm).mean()
        return lossvalue


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


class LocalizationNet(torch.nn.Module):
    """E2E module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options
    """

    def __init__(self, idim, odim, args):
        torch.nn.Module.__init__(self)
        self.outdir = args.outdir
        self.reporter = Reporter()
        self.loc_net = ConvNet(idim, 1, int(odim), args.arch_type)
        self.num_spkrs = 2
        self.pit = PIT(self.num_spkrs)
        self.loss = None
        self.loss_type = args.loss_type
        self.arch_type = args.arch_type
        if self.loss_type == 'ce':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif self.loss_type == 'emd':
            self.loss_func = EMD(classes=odim)
        elif self.loss_type == 'sce':
            self.loss_func = SCE(classes=odim)
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
        if self.arch_type != 3:
            hs = [None] * self.num_spkrs
            ys = [None] * self.num_spkrs
            hs[0], hs[1] = self.loc_net(xs_pad, ilens)
            ys[0] = ys_pad['a1']
            ys[1] = ys_pad['a2']
            loss_perm = torch.stack([self.loss_func(hs[i // self.num_spkrs],
                                                    ys[i % self.num_spkrs])
                                     for i in range(self.num_spkrs ** 2)], dim=0)
            self.loss, min_perm = self.pit.min_pit_sample(loss_perm)
        else:
            hs = self.loc_net(xs_pad, ilens)
            hs = torch.sigmoid(hs)
            true_dist = torch.zeros_like(hs)
            true_dist.scatter_(1, ys_pad['a1'].data.unsqueeze(1), 1)
            true_dist.scatter_(1, ys_pad['a2'].data.unsqueeze(1), 1)
            self.loss = self.loss_func(hs, true_dist)
        loss_data = float(self.loss)
        if not math.isnan(loss_data):
            self.reporter.report(loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def classify(self, xs, res_step=10):
        """Forwarding only the frontend stage

        :param ndarray xs: input acoustic feature (T, C, F)
        """

        prev = self.training
        self.eval()
        ys = [None] * self.num_spkrs
        ss = [None] * self.num_spkrs
        ms = [None] * self.num_spkrs
        fs = [None] * self.num_spkrs
        ws = [None] * self.num_spkrs
        ts = [None] * self.num_spkrs
        ys2 = [None] * self.num_spkrs
        angle = [None] * self.num_spkrs
        angle2 = [None] * self.num_spkrs
        xs_p = [x['p'] for x in xs]
        ys_1 = [x['angle1'] for x in xs]
        ys_2 = [x['angle2'] for x in xs]
        ms_1 = [x['mhyp1'] for x in xs]
        ms_2 = [x['mhyp2'] for x in xs]
        ss_1 = [x['shyp1'] for x in xs]
        ss_2 = [x['shyp2'] for x in xs]
        ws_1 = [x['whyp1'] for x in xs]
        ws_2 = [x['whyp2'] for x in xs]
        fs_1 = [x['fhyp1'] for x in xs]
        fs_2 = [x['fhyp2'] for x in xs]
        ts_1 = [x['thyp1'] for x in xs]
        ts_2 = [x['thyp2'] for x in xs]
        ilens = np.fromiter((xx.shape[0] for xx in xs_p), dtype=np.int64)
        xs_p = [to_device(self, to_torch_tensor(xx).float()) for xx in xs_p]
        xs_pad = pad_list(xs_p, 0.0)
        ilens = torch.from_numpy(np.asarray(ilens)).to(xs_pad.device)
        ys[0] = torch.from_numpy(np.asarray(ys_1)).long().to(xs_pad.device)
        ys[1] = torch.from_numpy(np.asarray(ys_2)).long().to(xs_pad.device)
        ys[0] = ys[0]/10
        ys[1] = ys[1]/10
        ms[0] = torch.from_numpy(np.asarray(ms_1)).long().to(xs_pad.device)
        ms[1] = torch.from_numpy(np.asarray(ms_2)).long().to(xs_pad.device)
        ms[0] = ms[0]/10
        ms[1] = ms[1]/10
        fs[0] = torch.from_numpy(np.asarray(fs_1)).long().to(xs_pad.device)
        fs[1] = torch.from_numpy(np.asarray(fs_2)).long().to(xs_pad.device)
        fs[0] = fs[0]/10
        fs[1] = fs[1]/10
        ts[0] = torch.from_numpy(np.asarray(ts_1)).long().to(xs_pad.device)
        ts[1] = torch.from_numpy(np.asarray(ts_2)).long().to(xs_pad.device)
        ts[0] = ts[0]/10
        ts[1] = ts[1]/10
        ws[0] = torch.from_numpy(np.asarray(ws_1)).long().to(xs_pad.device)
        ws[1] = torch.from_numpy(np.asarray(ws_2)).long().to(xs_pad.device)
        ws[0] = ws[0]/10
        ws[1] = ws[1]/10
        ss[0] = torch.from_numpy(np.asarray(ss_1)).long().to(xs_pad.device)
        ss[1] = torch.from_numpy(np.asarray(ss_2)).long().to(xs_pad.device)
        ss[0] = ss[0]/10
        ss[1] = ss[1]/10
        if self.arch_type != 3:
            hs = [None] * self.num_spkrs
            hs[0], hs[1] = self.loc_net(xs_pad, ilens)
            angle[0] = torch.max(hs[0],1)[1]
            angle[1] = torch.max(hs[1],1)[1]
        else:
            hs = self.loc_net(xs_pad, ilens)
            hs = torch.sigmoid(hs)
            angles = torch.topk(hs, k=2)
            angle[0] = angles[1][0][0]
            angle[1] = angles[1][0][1]
        if prev:
            self.train()
        loss_perm = torch.stack([-((angle[i // self.num_spkrs] == ys[i % self.num_spkrs]).float())
                                 for i in range(self.num_spkrs ** 2)], dim=1)
        ret = self.pit.min_pit_sample(loss_perm[0])
        if ret[1][0] == 0:
            angle1 = angle[0]
            angle2 = angle[1]
        else:
            angle1 = angle[1]
            angle2 = angle[0]
        loss_perm = torch.stack([-((ms[i // self.num_spkrs] == ys[i % self.num_spkrs]).float())
                                 for i in range(self.num_spkrs ** 2)], dim=1)
        ret_m = self.pit.min_pit_sample(loss_perm[0])
        if ret_m[1][0] == 0:
            m1 = ms[0]
            m2 = ms[1]
        else:
            m1 = ms[1]
            m2 = ms[0]
        loss_perm = torch.stack([-((ws[i // self.num_spkrs] == ys[i % self.num_spkrs]).float())
                                 for i in range(self.num_spkrs ** 2)], dim=1)
        ret_w = self.pit.min_pit_sample(loss_perm[0])
        if ret_w[1][0] == 0:
            w1 = ws[0]
            w2 = ws[1]
        else:
            w1 = ws[1]
            w2 = ws[0]
        loss_perm = torch.stack([-((ts[i // self.num_spkrs] == ys[i % self.num_spkrs]).float())
                                 for i in range(self.num_spkrs ** 2)], dim=1)
        ret_t = self.pit.min_pit_sample(loss_perm[0])
        if ret_t[1][0] == 0:
            t1 = ts[0]
            t2 = ts[1]
        else:
            t1 = ts[1]
            t2 = ts[0]
        loss_perm = torch.stack([-((fs[i // self.num_spkrs] == ys[i % self.num_spkrs]).float())
                                 for i in range(self.num_spkrs ** 2)], dim=1)
        ret_f = self.pit.min_pit_sample(loss_perm[0])
        if ret_f[1][0] == 0:
            f1 = fs[0]
            f2 = fs[1]
        else:
            f1 = fs[1]
            f2 = fs[0]
        loss_perm = torch.stack([-((ss[i // self.num_spkrs] == ys[i % self.num_spkrs]).float())
                                 for i in range(self.num_spkrs ** 2)], dim=1)
        ret_s = self.pit.min_pit_sample(loss_perm[0])
        if ret_s[1][0] == 0:
            s1 = ss[0]
            s2 = ss[1]
        else:
            s1 = ss[1]
            s2 = ss[0]
        return angle1.cpu().numpy(), angle2.cpu().numpy(), -ret[0].cpu().numpy(), -ret_m[0].cpu().numpy(), -ret_s[0].cpu().numpy(), -ret_w[0].cpu().numpy(), -ret_t[0].cpu().numpy(), -ret_f[0].cpu().numpy(), ys[0].cpu().numpy(), ys[1].cpu().numpy(), m1.cpu().numpy(), m2.cpu().numpy(), s1.cpu().numpy(), s2.cpu().numpy(), w1.cpu().numpy(), w2.cpu().numpy(), t1.cpu().numpy(), t2.cpu().numpy(), f1.cpu().numpy(), f2.cpu().numpy()


class ConvNet(torch.nn.Module):
    """VGG-like module

    :param int in_channel: number of input channels
    """

    def __init__(self, idim, in_channel=1, odim=360, arch_type=1):
        super(ConvNet, self).__init__()
        # CNN layer (VGG motivated)
        self.conv1_1 = torch.nn.Conv2d(in_channel, 8, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        dim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # max pooling
        self.linear_1 = torch.nn.Linear(int(dim) * 16 * 3, odim * 2)
        subsample = np.ones(2, dtype=np.int)
        self.arch_type=arch_type
        if arch_type==1 or arch_type==2:
            self.brnn = RNNP(odim*2, 1, odim*2, odim*2, subsample, dropout=0.0)
        if arch_type==3:
            self.linear_2_1 = torch.nn.Linear(odim * 2, odim)
            self.linear_ps_1 = torch.nn.Linear(odim, odim)
        elif arch_type==2:
            self.linear_2_1 = torch.nn.Linear(odim * 2, odim * 2)
            self.linear_2_2 = torch.nn.Linear(odim * 2, odim * 2)
            self.linear_ps_1 = torch.nn.Linear(odim * 2, odim)
            self.linear_ps_2 = torch.nn.Linear(odim * 2, odim)
        else:
            self.linear_2_1 = torch.nn.Linear(odim * 2, odim)
            self.linear_2_2 = torch.nn.Linear(odim * 2, odim)
            self.linear_ps_1 = torch.nn.Linear(odim, odim)
            self.linear_ps_2 = torch.nn.Linear(odim, odim)

        self.in_channel = in_channel

    def forward(self, xs_pad, ilens, eps=1e-15, **kwargs):
        """VGG2L forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of padded hidden state sequences (B, Tmax // 4, 128)
        :rtype: torch.Tensor
        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        # xs_pad = F.pad_sequence(xs_pad)

        B, input_length, C, Fd = xs_pad.size()
        xs_pad = xs_pad.view(-1, xs_pad.size(-2), xs_pad.size(-1))
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), self.in_channel,
                             xs_pad.size(2) // self.in_channel).transpose(1, 2)

        # NOTE: max_pool1d ?
        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)

        xs_pad = F.relu(self.conv2_1(xs_pad))
        #xs_pad = F.relu(self.conv2_2(xs_pad))

        xs = xs_pad.view(
            xs_pad.size(0), xs_pad.size(1) * xs_pad.size(2) * xs_pad.size(3))
        xs = xs.view(-1, input_length, xs.size(-1))
        xs = torch.relu(self.linear_1(xs))
        if self.arch_type==1:
            xs, _, _ = self.brnn(xs, ilens)
        if self.arch_type==3:
            xs_1 = torch.relu(self.linear_2_1(xs))
            mask_pad = to_device(self, make_pad_mask(ilens).unsqueeze(-1))
            xs_1_masked = xs_1.masked_fill(mask_pad, 0)
            xs_1 = torch.sum(xs_1_masked, dim=1)
            ilens.unsqueeze_(-1)
            xs_1 = xs_1/ilens.float()
            xs_1 = self.linear_ps_1(xs_1)
            return xs_1
        elif self.arch_type==2:
            xs_m, _, _ = self.brnn(xs, ilens)
            mask_1 = self.linear_2_1(xs_m)
            mask_2 = self.linear_2_2(xs_m)
            mask_1 = torch.sigmoid(mask_1)
            mask_2 = torch.sigmoid(mask_2)
            mask_1.masked_fill(make_pad_mask(ilens, mask_1, length_dim=1), 0)
            mask_2.masked_fill(make_pad_mask(ilens, mask_2, length_dim=1), 0)
            mask_1 = mask_1 / (mask_1.sum(dim=1, keepdim=True) + eps)
            mask_2 = mask_2 / (mask_2.sum(dim=1, keepdim=True) + eps)
            xs_1_masked = xs * mask_1
            xs_2_masked = xs * mask_2
            xs_1 = torch.sum(xs_1_masked, dim=1)
            xs_2 = torch.sum(xs_2_masked, dim=1)
            xs_1 = self.linear_ps_1(xs_1)
            xs_2 = self.linear_ps_2(xs_2)
            return xs_1, xs_2
        else:
            xs_1 = torch.relu(self.linear_2_1(xs))
            xs_2 = torch.relu(self.linear_2_2(xs))
            mask_pad = to_device(self, make_pad_mask(ilens).unsqueeze(-1))
            xs_1_masked = xs_1.masked_fill(mask_pad, 0)
            xs_2_masked = xs_2.masked_fill(mask_pad, 0)
            xs_1 = torch.sum(xs_1_masked, dim=1)
            xs_2 = torch.sum(xs_2_masked, dim=1)
            ilens.unsqueeze_(-1)
            xs_1 = xs_1/ilens.float()
            xs_2 = xs_2/ilens.float()
            xs_1 = self.linear_ps_1(xs_1)
            xs_2 = self.linear_ps_2(xs_2)
            return xs_1, xs_2
