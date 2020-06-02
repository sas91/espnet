from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pudb
import numpy
import torch
import torch.nn as nn
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.frontends.dnn_beamformer import DNN_Beamformer
from espnet.nets.pytorch_backend.frontends.dnn_wpe import DNN_WPE
from espnet.nets.pytorch_backend.frontends.wpe import wpe


class Frontend(nn.Module):
    def __init__(self,
                 idim: int,
                 # WPE options
                 use_wpe: bool = False,
                 use_wpe_test: bool = False,
                 wtype: str = 'blstmp',
                 wlayers: int = 3,
                 wunits: int = 300,
                 wprojs: int = 320,
                 wdropout_rate: float = 0.0,
                 taps: int = 5,
                 delay: int = 3,
                 use_dnn_mask_for_wpe: bool = True,

                 # Beamformer options
                 use_beamformer: bool = False,
                 btype: str = 'blstmp',
                 bftype: str = 'mvdr',
                 blayers: int = 3,
                 bunits: int = 300,
                 bprojs: int = 320,
                 badim: int = 320,
                 ref_channel: int = -1,
                 bdropout_rate=0.0):
        super().__init__()

        self.use_beamformer = use_beamformer
        self.use_wpe = use_wpe
        self.use_dnn_mask_for_wpe = use_dnn_mask_for_wpe

        if self.use_wpe:
            if self.use_dnn_mask_for_wpe:
                # Use DNN for power estimation
                # (Not observed significant gains)
                iterations = 1
            else:
                # Performing as conventional WPE, without DNN Estimator
                iterations = 2

            self.wpe = DNN_WPE(wtype=wtype,
                               widim=idim,
                               wunits=wunits,
                               wprojs=wprojs,
                               wlayers=wlayers,
                               taps=taps,
                               delay=delay,
                               dropout_rate=wdropout_rate,
                               iterations=iterations,
                               use_dnn_mask=use_dnn_mask_for_wpe)
        #elif self.use_wpe_test:
        #    iterations = 2
        #    self.wpe = DNN_WPE(taps=taps,
        #                       delay=delay,
        #                       iterations=iterations,
        #                       use_dnn_mask=False)
        else:
            self.wpe = None

        if self.use_beamformer:
            self.beamformer = DNN_Beamformer(btype=btype,
                                             bidim=idim,
                                             bunits=bunits,
                                             bprojs=bprojs,
                                             blayers=blayers,
                                             dropout_rate=bdropout_rate,
                                             badim=badim,
                                             ref_channel=ref_channel,
                                             beamformer_type=bftype)
        else:
            self.beamformer = None

    def forward(self, x, ilens, sv=None, af=None, xv=None, use_wpe_test=False,
                use_bf_test=False, ss=None, ilens_ss=None):
        assert len(x) == len(ilens), (len(x), len(ilens))
        # (B, T, F) or (B, T, C, F)
        if x.dim() not in (3, 4):
            raise ValueError(f'Input dim must be 3 or 4: {x.dim()}')
        if not torch.is_tensor(ilens):
            ilens = torch.from_numpy(numpy.asarray(ilens)).to(x.device)
        if ss is not None:
            if not torch.is_tensor(ilens_ss):
                ilens_ss = torch.from_numpy(numpy.asarray(ilens_ss)).to(x.device)

        mask = None
        mask_speech = None
        mask_noise = None
        mask_interference = None
        mask_post = None
        sigma = None
        se = None
        h = x
        if h.dim() == 4 or af is not None or xv is not None:
            if self.training:
                choices=[]
                #choices = [(False, False)]
                if self.use_wpe:
                    choices.append((True, False))

                if self.use_beamformer:
                    choices.append((False, True))

                use_wpe, use_beamformer = \
                    choices[numpy.random.randint(len(choices))]
                use_wpe_test = False
            else:
                use_wpe = self.use_wpe
                use_beamformer = self.use_beamformer

            # 1. WPE
            if use_wpe:
                # h: (B, T, C, F) -> h: (B, T, C, F)
                h, ilens, mask = self.wpe(h, ilens)

            if use_wpe_test:
                # h: (B, T, C, F) -> h: (B, T, C, F)
                enhanced = h.permute(0, 3, 2, 1)
                enhanced = wpe(enhanced)
                h = enhanced.permute(0, 3, 2, 1)

            # 2. Beamformer
            if use_beamformer:
                # h: (B, T, C, F) -> h: (B, T, F)
                h, ilens, mask_speech, mask_noise, mask_interference, mask_post, sigma, se = self.beamformer(h, ilens, af, sv, xv, ss, ilens_ss, use_bf_test)

        return h, ilens, mask_speech, mask_noise, mask_interference, mask_post, sigma, se


def frontend_for(args, idim):
    return Frontend(
        idim=idim,
        # WPE options
        use_wpe=args.use_wpe,
        use_wpe_test=args.use_wpe_test,
        wtype=args.wtype,
        wlayers=args.wlayers,
        wunits=args.wunits,
        wprojs=args.wprojs,
        wdropout_rate=args.wdropout_rate,
        taps=args.wpe_taps,
        delay=args.wpe_delay,
        use_dnn_mask_for_wpe=args.use_dnn_mask_for_wpe,

        # Beamformer options
        use_beamformer=args.use_beamformer,
        btype=args.btype,
        bftype=args.bftype,
        blayers=args.blayers,
        bunits=args.bunits,
        bprojs=args.bprojs,
        badim=args.badim,
        ref_channel=args.ref_channel,
        bdropout_rate=args.bdropout_rate)
