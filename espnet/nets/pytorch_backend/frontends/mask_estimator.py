from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet.nets.pytorch_backend.rnn.encoders import RNNP
from espnet.nets.pytorch_backend.nets_utils import to_device
import pudb

class MaskEstimator(torch.nn.Module):
    def __init__(self, type, idim, layers, units, projs, dropout, nmask=1,
                 cd_mask=False, SA=False, SA_type='aux', SA_dim=30, odim=0,
                 xvdim=128, post=False, use_ipd=False, only_mask=False):
        super().__init__()
        subsample = np.ones(layers + 1, dtype=np.int)

        typ = type.lstrip("vgg").rstrip("p")
        if type[-1] == "p":
            if SA and (SA_type == 'sb' or SA_type == 'sb_ss'):
                self.brnn_pre = RNNP(idim, 1, units, idim, subsample, dropout,
                                     typ=typ)
                self.brnn = RNNP(idim, layers, units, projs, subsample, dropout,
                                 typ=typ)
            elif SA_type == 'lhuc_ss_attention':
                self.brnn_pre = RNNP(idim, 1, idim, idim, subsample, dropout,
                                     typ=typ)
                self.brnn_ss = RNNP(idim, 1, idim, idim, subsample, dropout,
                                     typ=typ)
                self.brnn = RNNP(idim, layers, units, projs, subsample, dropout,
                                 typ=typ)
                self.linear_sa_lhuc = torch.nn.Linear(idim, idim)
            elif SA_type == 'lhuc_ss_attention_2':
                self.brnn = RNNP(idim, layers, units, projs, subsample, dropout,
                                 typ=typ)
                self.linear_ss_1 = torch.nn.Linear(odim, odim)
                self.linear_ss_2 = torch.nn.Linear(odim, odim)
                self.linear_pre_1 = torch.nn.Linear(odim, odim)
                self.linear_pre_2 = torch.nn.Linear(odim, odim)
            elif SA and ('lhuc' in SA_type or 'cat' in SA_type):
                if use_ipd:
                    self.brnn_pre = RNNP(idim, 1, units, projs, subsample, dropout,
                                         typ=typ)
                    self.brnn = RNNP(projs, layers-1, units, projs, subsample, dropout,
                                     typ=typ)
                else:
                    self.brnn_pre = RNNP(idim, 1, units, idim, subsample, dropout,
                                         typ=typ)
                    if 'cat' in SA_type:
                        self.brnn = RNNP(idim+xvdim, layers, units, projs, subsample, dropout,
                                         typ=typ)
                    else:
                        self.brnn = RNNP(idim, layers, units, projs, subsample, dropout,
                                         typ=typ)
            else:
                self.brnn = RNNP(idim, layers, units, projs, subsample, dropout,
                                 typ=typ)
        else:
            self.brnn = RNN(idim, layers, units, projs, dropout, typ=typ)

        self.type = type
        self.nmask = nmask
        self.cd_mask = cd_mask
        self.post = post
        self.only_mask = only_mask
        self.SA = SA
        self.SA_type = SA_type
        self.SA_dim = SA_dim
        self.use_ipd = use_ipd
        if SA and 'lhuc_ss_attention' not in SA_type:
            if use_ipd:
                if SA_type=='lhuc' or SA_type=='sb' or SA_type=='aux':
                    self.linear_sa = torch.nn.Linear(xvdim, projs)
                elif SA_type=='lhuc_ss' or SA_type=='lhuc_ss_cd':
                    self.linear_sa = torch.nn.Linear(projs, projs)
                if 'lhuc' in SA_type:
                    self.linear_sa_lhuc_1 = torch.nn.Linear(odim, projs)
                    self.linear_sa_lhuc_2 = torch.nn.Linear(projs, projs)
            else:
                if SA_type=='lhuc' or SA_type=='sb' or SA_type=='aux':
                    self.linear_sa = torch.nn.Linear(xvdim, idim)
                elif 'lhuc_ss' in SA_type or SA_type=='sb_ss' or SA_type=='aux_ss':
                    self.linear_sa = torch.nn.Linear(idim, idim)
                if SA_type=='sb' or SA_type=='sb_ss':
                    self.linear_sa_sb = torch.nn.Linear(idim, SA_dim)
                    self.linears_sal = torch.nn.ModuleList(
                        [torch.nn.Linear(idim, idim) for _ in range(SA_dim)])
                if 'lhuc' in SA_type or SA_type=='sb_ss':
                    self.linear_sa_lhuc_1 = torch.nn.Linear(idim, idim)
                    self.linear_sa_lhuc_2 = torch.nn.Linear(idim, idim)
        if cd_mask or use_ipd:
            self.linears = torch.nn.ModuleList(
                [torch.nn.Linear(projs, odim) for _ in range(nmask)])
        else:
            self.linears = torch.nn.ModuleList(
                [torch.nn.Linear(projs, idim) for _ in range(nmask)])

    def forward(self, xs, ilens, af=None, xv=None, ss=None, ilens_ss=None, use_bf_test=False):
        """The forward function

        Args:
            xs: (B, F, C, T)
            ilens: (B,)
        Returns:
            hs (torch.Tensor): The hidden vector (B, F, C, T)
            masks: A tuple of the masks. (B, F, C, T)
            ilens: (B,)
        """
        assert xs.size(0) == ilens.size(0), (xs.size(0), ilens.size(0))
        if self.post or (self.only_mask and not use_bf_test) or self.use_ipd or (self.SA and 'cd' not in self.SA_type):
            _, input_length, _ = xs.size()
        else:
            B, Fd, C, input_length = xs.size()
        se = None
        if self.post:
            xs = (xs.real ** 2 + xs.imag ** 2) ** 0.5
        #if self.SA:
        #    xv = self.linear_sa(xv)
        #    # (B, F, C, T) -> (B, C, T, F)
        #    xs = xs.permute(0, 2, 3, 1)
        #    # Calculate amplitude: (B, C, T, F) -> (B, C, T, F)
        #    #xs = (xs.real ** 2 + xs.imag ** 2) ** 0.5
        #    if self.SA_type=='aux':
        #        xv.unsqueeze_(-1).unsqueeze_(-1)
        #        xv = xv.permute(0, 2, 3, 1)
        #        xs = xs + xv
        #    elif self.SA_type=='sb':
        #        xv = self.linear_sa_sb(xv)
        #        xv = xv.permute(1, 0)
        #        xv.unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1)
        #        #xv = F.softmax(xv, dim=0)
        #        # xs: (B, C, T, F) -> xs: (B * C, T, F)
        #        xs = xs.view(-1, xs.size(-2), xs.size(-1))
        #        # ilens: (B,) -> ilens_: (B * C)
        #        ilens_ = ilens[:, None].expand(-1, C).contiguous().view(-1)
        #        # xs: (B * C, T, F) -> xs: (B * C, T, F)
        #        xs, _, _ = self.brnn_pre(xs, ilens_)
        #        # xs: (B * C, T, F) -> xs: (B, C, T, F)
        #        xs = xs.view(-1, C, xs.size(-2), xs.size(-1))
        #        i = 0
        #        xs_l = []
        #        for linear in self.linears_sal:
        #            temp = linear(xs)
        #            xs_l.append(linear(xs))
        #            i = i+1
        #        xs_v = torch.stack(xs_l, dim=0)
        #        xs_v = xs_v * xv
        #        xs = torch.sum(xs_v, dim=0)
        #    # xs: (B, C, T, F) -> xs: (B * C, T, F)
        #    xs = xs.view(-1, xs.size(-2), xs.size(-1))
        #    # ilens: (B,) -> ilens_: (B * C)
        #    ilens_ = ilens[:, None].expand(-1, C).contiguous().view(-1)
        #    # xs: (B * C, T, F) -> xs: (B * C, T, D)
        #    xs, _, _ = self.brnn(xs, ilens_)
        #    # xs: (B * C, T, D) -> xs: (B, C, T, D)
        #    xs = xs.view(-1, C, xs.size(-2), xs.size(-1))
        if self.SA and not self.only_mask:
            if self.SA_type=='aux':
                xv = self.linear_sa(xv)
                xv.unsqueeze_(-1)
                xv = xv.permute(0, 2, 1)
                xs = xs.permute(0, 2, 1)
                xs = xs + xv
            elif self.SA_type=='lhuc':
                xs = xs.permute(0,2,1)
                xs, _, _ = self.brnn_pre(xs, ilens)
                xv = self.linear_sa(xv)
                xv = torch.relu(self.linear_sa_lhuc_1(torch.relu(xv)))
                xv = self.linear_sa_lhuc_2(xv)
                xv.unsqueeze_(-1)
                xv = xv.permute(0, 2, 1)
                xs = xs * xv
            elif self.SA_type=='lhuc_ss_attention':
                if self.use_ipd:
                    af = af.permute(0, 3, 2, 1)
                    xs = torch.cat((xs[:, :, None],af),2)
                    xs = xs.view(xs.size(0), -1, xs.size(3))
                    # (B, F*C, T) -> (B, T, F*C)
                xs = xs.permute(0, 2, 1)
                xs, _, _ = self.brnn_pre(xs, ilens)
                ilens_ss_sorted, indices = torch.sort(ilens_ss,
                                                      descending=True)
                ss_sorted = ss[indices]
                ss_sorted, _, _ = self.brnn_ss(ss_sorted, ilens_ss_sorted)
                ss = ss_sorted.new(*ss_sorted.size())
                ss.scatter_(0, indices.unsqueeze(1).unsqueeze(1).expand(-1, ss.shape[1], ss.shape[2]), ss_sorted)
                ss_t = ss.transpose(1,2)
                score=torch.matmul(xs,ss_t)
                #mask_pad = to_device(self, make_pad_mask(ilens).unsqueeze(-1))
                #score_masked = score.masked_fill(mask_pad, -float('inf'))
                #mask_pad = to_device(self, make_pad_mask(ilens_ss).unsqueeze(-1))
                #score_masked = score_masked.transpose(1,2)
                #score_masked_2 = score_masked.masked_fill(mask_pad, -float('inf'))
                #score_masked_2 = score_masked_2.transpose(1,2)
                #align=torch.softmax(score_masked_2, dim=2)
                mask_pad = to_device(self, make_pad_mask(ilens_ss).unsqueeze(-1))
                score = score.transpose(1,2)
                score_masked = score.masked_fill(mask_pad, -float('inf'))
                score_masked = score_masked.transpose(1,2)
                align=torch.softmax(score_masked/100, dim=2)
                #weight=align.unsqueeze(-1)
                #weight=weight.permute(1,0,2,3)
                se=torch.matmul(align,ss)
                se = torch.sigmoid(self.linear_sa_lhuc(se))
                xs = xs * se
            elif self.SA_type=='lhuc_ss_attention_2':
                if self.use_ipd:
                    af = af.permute(0, 3, 2, 1)
                    xs = torch.cat((xs[:, :, None],af),2)
                    xs = xs.view(xs.size(0), -1, xs.size(3))
                    # (B, F*C, T) -> (B, T, F*C)
                xs = xs.permute(0, 2, 1)
                ss = torch.relu(self.linear_ss_1(ss))
                ss = torch.relu(self.linear_ss_2(ss))
                xs = torch.relu(self.linear_pre_1(xs))
                xs = torch.relu(self.linear_pre_2(xs))
                ss_t = ss.transpose(1,2)
                score=torch.matmul(xs,ss_t)
                #mask_pad = to_device(self, make_pad_mask(ilens).unsqueeze(-1))
                #score_masked = score.masked_fill(mask_pad, -float('inf'))
                #mask_pad = to_device(self, make_pad_mask(ilens_ss).unsqueeze(-1))
                #score_masked = score_masked.transpose(1,2)
                #score_masked_2 = score_masked.masked_fill(mask_pad, -float('inf'))
                #score_masked_2 = score_masked_2.transpose(1,2)
                #align=torch.softmax(score_masked_2, dim=2)
                mask_pad = to_device(self, make_pad_mask(ilens_ss).unsqueeze(-1))
                score = score.transpose(1,2)
                score_masked = score.masked_fill(mask_pad, -float('inf'))
                score_masked = score_masked.transpose(1,2)
                align=torch.softmax(score_masked, dim=2)
                #weight=align.unsqueeze(-1)
                #weight=weight.permute(1,0,2,3)
                se=torch.matmul(align,ss)
                xs = xs * se
            elif self.SA_type=='lhuc_ss':
                if self.use_ipd:
                    af = af.permute(0, 3, 2, 1)
                    xs = torch.cat((xs[:, :, None],af),2)
                    xs = xs.view(xs.size(0), -1, xs.size(3))
                    # (B, F*C, T) -> (B, T, F*C)
                xs = xs.permute(0, 2, 1)
                xs, _, _ = self.brnn_pre(xs, ilens)
                ss = torch.relu(self.linear_sa_lhuc_1(ss))
                ss = torch.relu(self.linear_sa_lhuc_2(ss))
                ss = self.linear_sa(ss)
                mask_pad = to_device(self, make_pad_mask(ilens_ss).unsqueeze(-1))
                ss_masked = ss.masked_fill(mask_pad, 0)
                ss = torch.sum(ss_masked, dim=1)
                ilens_ss.unsqueeze_(-1)
                ss = ss/ilens_ss.float()
                se = ss
                ss.unsqueeze_(-1)
                ss = ss.permute(0, 2, 1)
                xs = xs * ss
            elif self.SA_type=='lhuc_ss_cd':
                # (B, F, C, T) -> (B, C, T, F)
                xs = xs.permute(0, 2, 3, 1)
                # Calculate amplitude: (B, C, T, F) -> (B, C, T, F)
                xs = (xs.real ** 2 + xs.imag ** 2) ** 0.5
                # xs: (B, C, T, F) -> xs: (B * C, T, F)
                xs = xs.view(-1, xs.size(-2), xs.size(-1))
                # ilens: (B,) -> ilens_: (B * C)
                ilens_ = ilens[:, None].expand(-1, C).contiguous().view(-1)
                xs, _, _ = self.brnn_pre(xs, ilens_)
                ss = torch.relu(self.linear_sa_lhuc_1(ss))
                ss = torch.relu(self.linear_sa_lhuc_2(ss))
                ss = self.linear_sa(ss)
                mask_pad = to_device(self, make_pad_mask(ilens_ss).unsqueeze(-1))
                ss_masked = ss.masked_fill(mask_pad, 0)
                ss = torch.sum(ss_masked, dim=1)
                ilens_ss.unsqueeze_(-1)
                ss = ss/ilens_ss.float()
                se = ss
                ss.unsqueeze_(-1).unsqueeze_(-1)
                ss = ss.permute(0, 2, 3, 1)
                xs = xs.view(-1, C, xs.size(-2), xs.size(-1))
                xs = xs * ss
                xs = xs.view(-1, xs.size(-2), xs.size(-1))
                # xs: (B * C, T, F) -> xs: (B * C, T, D)
                xs, _, _ = self.brnn(xs, ilens_)
                # xs: (B * C, T, D) -> xs: (B, C, T, D)
                xs = xs.view(-1, C, xs.size(-2), xs.size(-1))
            elif self.SA_type=='lhuc_cd':
                xs = xs.permute(0, 2, 3, 1)
                xs = (xs.real ** 2 + xs.imag ** 2) ** 0.5
                xs = xs.view(-1, xs.size(-2), xs.size(-1))
                xs, _, _ = self.brnn_pre(xs, ilens)
                xv = self.linear_sa(xv)
                xv = torch.relu(self.linear_sa_lhuc_1(torch.relu(xv)))
                xv = self.linear_sa_lhuc_2(xv)
                xv.unsqueeze_(-1).unsqueeze_(-1)
                xv = xv.permute(0, 2, 3, 1)
                xv = xv.view(-1, C, xs.size(-2), xs.size(-1))
                xs = xs * xv
                xs = xs.view(-1, xs.size(-2), xs.size(-1))
                # xs: (B * C, T, F) -> xs: (B * C, T, D)
                xs, _, _ = self.brnn(xs, ilens_)
                # xs: (B * C, T, D) -> xs: (B, C, T, D)
                xs = xs.view(-1, C, xs.size(-2), xs.size(-1))
            elif self.SA_type=='sb_ss':
                ss = torch.relu(self.linear_sa_lhuc_1(ss))
                ss = torch.relu(self.linear_sa_lhuc_2(ss))
                ss = self.linear_sa(ss)
                mask_pad = to_device(self, make_pad_mask(ilens_ss).unsqueeze(-1))
                ss_masked = ss.masked_fill(mask_pad, 0)
                ss = torch.sum(ss_masked, dim=1)
                ilens_ss.unsqueeze_(-1)
                ss = ss/ilens_ss.float()
                ss = self.linear_sa_sb(ss)
                ss = ss.permute(1, 0)
                ss.unsqueeze_(-1).unsqueeze_(-1)
                #ss = F.softmax(ss, dim=0)
                se = ss
                xs = xs.permute(0,2,1)
                xs, _, _ = self.brnn_pre(xs, ilens)
                i = 0
                xs_l = []
                for linear in self.linears_sal:
                    xs_l.append(linear(xs))
                    i = i+1
                xs_v = torch.stack(xs_l, dim=0)
                xs_v = xs_v * ss
                xs = torch.sum(xs_v, dim=0)
            elif self.SA_type=='sb':
                xv = self.linear_sa(xv)
                xv = self.linear_sa_sb(xv)
                xv = xv.permute(1, 0)
                xv.unsqueeze_(-1).unsqueeze_(-1)
                #xv = F.softmax(xv, dim=0)
                #xv = torch.relu(xv)
                xs = xs.permute(0,2,1)
                xs, _, _ = self.brnn_pre(xs, ilens)
                i = 0
                xs_l = []
                for linear in self.linears_sal:
                    xs_l.append(linear(xs))
                    i = i+1
                xs_v = torch.stack(xs_l, dim=0)
                xs_v = xs_v * xv
                xs = torch.sum(xs_v, dim=0)
            if self.SA_type!='lhuc_ss_cd':
                xs, _, _ = self.brnn(xs, ilens)
        elif self.use_ipd:
            af = af.permute(0, 3, 2, 1)
            # Calculate amplitude: (B, C, T, F) -> (B, C, T, F)
            #xs = (xs[:,:,0,:].real ** 2 + xs[:,:,0,:].imag ** 2) ** 0.5
            #xs.unsqueeze_(-1)
            #xs = xs.permute(0, 1, 3, 2)
            #xs = torch.cat((xs, af), 2)
            xs = torch.cat((xs[:, :, None],af),2)
            ilens_ = ilens
            # (B, F, C, T) -> (B, F*C, T)
            xs = xs.view(xs.size(0), -1, xs.size(3))
            # (B, F*C, T) -> (B, T, F*C)
            xs = xs.permute(0, 2, 1)
            # xs: (B, T, F*C) -> xs: (B, T, D)
            xs, _, _ = self.brnn(xs, ilens_)
        elif self.cd_mask:
            af=af.permute(0, 3, 2, 1)
            xs = torch.cat((xs.real, xs.imag, af.real, af.imag), 2)
            ilens_ = ilens
            # (B, F, C, T) -> (B, F*C, T)
            xs = xs.view(xs.size(0), -1, xs.size(3))
            # (B, F*C, T) -> (B, T, F*C)
            xs = xs.permute(0, 2, 1)
            # xs: (B, T, F*C) -> xs: (B, T, D)
            xs, _, _ = self.brnn(xs, ilens_)
        elif self.post or (self.only_mask and not use_bf_test):
                # xs: (B, T, F)
                # xs: (B, T, F) -> xs: (B, T, D)
                ilens_ = ilens
                if self.SA and self.SA_type=='aux':
                    xv = self.linear_sa(xv)
                    xv.unsqueeze_(-1)
                    xv = xv.permute(0, 2, 1)
                    xs = xs + xv
                if self.SA and self.SA_type=='cat':
                    xs, _, _ = self.brnn_pre(xs, ilens)
                    xv = F.normalize(xv, dim=0).unsqueeze(1).expand(-1,xs.size(1),-1)
                    xs = torch.cat([xs, xv], dim=-1)
                if self.SA and self.SA_type=='lhuc':
                    xs, _, _ = self.brnn_pre(xs, ilens)
                    xv = self.linear_sa(xv)
                    xv = torch.relu(self.linear_sa_lhuc_1(torch.relu(xv)))
                    xv = self.linear_sa_lhuc_2(xv)
                    xv.unsqueeze_(-1)
                    xv = xv.permute(0, 2, 1)
                    xs = xs * xv
                if self.SA and self.SA_type=='lhuc_ss':
                    xs, _, _ = self.brnn_pre(xs, ilens)
                    ss = torch.relu(self.linear_sa_lhuc_1(ss))
                    ss = torch.relu(self.linear_sa_lhuc_2(ss))
                    ss = self.linear_sa(ss)
                    mask_pad = to_device(self, make_pad_mask(ilens_ss).unsqueeze(-1))
                    ss_masked = ss.masked_fill(mask_pad, 0)
                    ss = torch.sum(ss_masked, dim=1)
                    ilens_ss.unsqueeze_(-1)
                    ss = ss/ilens_ss.float()
                    se = ss
                    ss.unsqueeze_(-1)
                    ss = ss.permute(0, 2, 1)
                    xs = xs * ss
                xs, _, _ = self.brnn(xs, ilens_)
        else:
            # (B, F, C, T) -> (B, C, T, F)
            xs = xs.permute(0, 2, 3, 1)
            # Calculate amplitude: (B, C, T, F) -> (B, C, T, F)
            xs = (xs.real ** 2 + xs.imag ** 2) ** 0.5
            # xs: (B, C, T, F) -> xs: (B * C, T, F)
            xs = xs.view(-1, xs.size(-2), xs.size(-1))
            # ilens: (B,) -> ilens_: (B * C)
            ilens_ = ilens[:, None].expand(-1, C).contiguous().view(-1)
            # xs: (B * C, T, F) -> xs: (B * C, T, D)
            xs, _, _ = self.brnn(xs, ilens_)
            # xs: (B * C, T, D) -> xs: (B, C, T, D)
            xs = xs.view(-1, C, xs.size(-2), xs.size(-1))

        masks = []
        for linear in self.linears:
            # xs: (B, C, T, D) -> mask:(B, C, T, F)
            # xs if CD: (B, T, D) -> mask:(B, T, F)
            mask = linear(xs)

            if self.cd_mask or self.post  or (self.only_mask and not use_bf_test) or self.use_ipd or (self.SA and self.SA_type!='lhuc_ss_cd'):
                # Zero padding
                mask = torch.sigmoid(mask)
                mask.masked_fill(make_pad_mask(ilens, mask, length_dim=1), 0)
                #mask = torch.clamp(mask, min=0, max=1)
                # (B, T, F) -> (B, F, T)
                if not self.only_mask:
                    mask = mask.permute(0, 2, 1)
            else:
                # Zero padding
                mask = torch.sigmoid(mask)
                mask.masked_fill(make_pad_mask(ilens, mask, length_dim=2), 0)
                # (B, C, T, F) -> (B, F, C, T)
                mask = mask.permute(0, 3, 1, 2)

            # Take cares of multi gpu cases: If input_length > max(ilens)
            #if mask.size(-1) < input_length:
            #    mask = F.pad(mask, [0, input_length - mask.size(-1)], value=0)
            masks.append(mask)

        return tuple(masks), ilens, se
