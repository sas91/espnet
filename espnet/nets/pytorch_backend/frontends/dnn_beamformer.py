from typing import Tuple

import torch
from torch.nn import functional as F

from espnet.nets.pytorch_backend.frontends.beamformer \
    import apply_beamforming_vector
from espnet.nets.pytorch_backend.frontends.beamformer \
    import get_mvdr_vector
from espnet.nets.pytorch_backend.frontends.beamformer \
    import get_smvdr_vector
from espnet.nets.pytorch_backend.frontends.beamformer \
    import get_gdr_vector
from espnet.nets.pytorch_backend.frontends.beamformer \
    import get_lcmv_vector
from espnet.nets.pytorch_backend.frontends.beamformer \
    import get_power_spectral_density_matrix
from espnet.nets.pytorch_backend.frontends.mask_estimator import MaskEstimator
from espnet.nets.pytorch_backend.frontends.mag_estimator import MagEstimator
from torch_complex.tensor import ComplexTensor
import pudb

class DNN_Beamformer(torch.nn.Module):
    """DNN mask based Beamformer

    Citation:
        Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;
        https://arxiv.org/abs/1703.04783

    """

    def __init__(self,
                 bidim,
                 btype='blstmp',
                 blayers=3,
                 bunits=300,
                 bprojs=320,
                 dropout_rate=0.0,
                 badim=320,
                 ref_channel: int = -1,
                 n_channel: int = 6,
                 beamformer_type='mvdr'):
        super().__init__()
        if beamformer_type != 'mvdr':
            if beamformer_type == 'mvdr_xv' or beamformer_type == 'mvdr_xv_pm':
                self.mask = MaskEstimator(btype, bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=2, cd_mask=False,
                                          SA=True, SA_type='lhuc_cd', odim=bidim)
                if beamformer_type == 'mvdr_xv_pm':
                    self.post_mask = MaskEstimator(btype, bidim,
                                                   1, bunits, bprojs,
                                                   dropout_rate, nmask=1, post=True)
            elif beamformer_type == 'mvdr_xv_sb':
                self.mask = MaskEstimator(btype, bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=3, cd_mask=False,
                                          SA=True, SA_type='sb', odim=bidim)
            elif beamformer_type == 'mvdr_angle' or beamformer_type =='mvdr_angle_post':
                self.mask = MaskEstimator(btype, ((2*n_channel)+3)*bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=2, odim=bidim,
                                          use_ipd=True)
            elif beamformer_type == 'mvdr_angle_mimo':
                self.mask = MaskEstimator(btype, ((2*n_channel)+3)*bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=3, odim=bidim,
                                          use_ipd=True)
            elif beamformer_type == 'mvdr_mimo':
                self.mask = MaskEstimator(btype, bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=3, cd_mask=False,
                                          odim=bidim)
            elif beamformer_type == 'gdr' or beamformer_type == 'gdr_pm' or beamformer_type == 'sb_mvdr':
                self.mask = MaskEstimator(btype, ((2*n_channel)+3)*bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=3, odim=bidim,
                                          use_ipd=True)
                if beamformer_type == 'gdr' or beamformer_type == 'gdr_pm':
                    self.sdw = SDW_net(bidim)
                if beamformer_type == 'gdr_pm':
                    self.post_mask = MaskEstimator(btype, bidim,
                                                   1, bunits, bprojs,
                                                   dropout_rate, nmask=1, post=True)
            elif beamformer_type == 'gdr2':
                self.mask = MaskEstimator(btype, ((2*n_channel)+3)*bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=2, odim=bidim,
                                          use_ipd=True)
                self.sdw = SDW_net_2(bidim, two_dim=False)
            elif beamformer_type == 'gdr3':
                self.mask = MaskEstimator(btype, ((2*n_channel)+3)*bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=2, odim=bidim,
                                          use_ipd=True)
                self.sdw = SDW_net_2(bidim)
            elif beamformer_type == 'mvdr_ss':
                self.mask = MaskEstimator(btype, bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=2, cd_mask=False,
                                          SA=True, SA_type='lhuc_ss_cd', odim=bidim)
            elif beamformer_type == 'gdr_angle_ss':
                self.mask = MaskEstimator(btype, ((2*n_channel)+3)*bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=3, cd_mask=False,
                                          SA=True, SA_type='lhuc_ss',
                                          odim=bidim, use_ipd=True)
                self.sdw = SDW_net(bidim)
            elif beamformer_type == 'mvdr_angle_ss' or beamformer_type == 'mvdr_angle_ss_post':
                self.mask = MaskEstimator(btype, ((2*n_channel)+3)*bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=2, cd_mask=False,
                                          SA=True, SA_type='lhuc_ss',
                                          odim=bidim, use_ipd=True)
            elif beamformer_type == 'mvdr_angle_ss_pm':
                self.mask = MaskEstimator(btype, ((2*n_channel)+3)*bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=2, odim=bidim,
                                          use_ipd=True)
                self.post_mask = MaskEstimator(btype, bidim,
                                               1, bunits, bprojs,
                                               dropout_rate, nmask=1, cd_mask=False,
                                               SA=True, SA_type='lhuc_ss',
                                               odim=bidim, post=True)
            elif beamformer_type == 'lcmv':
                self.mask = MaskEstimator(btype, ((2*n_channel)+3)*bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=1, odim=bidim,
                                          use_ipd=True)
            elif beamformer_type == 'mask_asr' or beamformer_type == 'smvdr':
                self.mask = MaskEstimator(btype, ((2*n_channel)+3)*bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=1, odim=bidim,
                                          use_ipd=True)
            elif beamformer_type == 'mask':
                self.mask = MaskEstimator(btype, ((2*n_channel)+3)*bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=1, odim=bidim,
                                          use_ipd=True)
            elif beamformer_type == 'mask_xv':
                self.mask = MaskEstimator(btype, bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=1, cd_mask=False,
                                          SA=True, SA_type='lhuc', odim=bidim)
            elif beamformer_type == 'mask_sb_ss':
                self.mask = MaskEstimator(btype, bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=1, cd_mask=False,
                                          SA=True, SA_type='sb_ss', odim=bidim)
            elif beamformer_type == 'mask_ss':
                self.mask = MaskEstimator(btype, bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=1, cd_mask=False,
                                          SA=True, SA_type='lhuc_ss', odim=bidim)
            elif beamformer_type == 'mask_ss_attention':
                self.mask = MaskEstimator(btype, bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=1, cd_mask=False,
                                          SA=True, SA_type='lhuc_ss_attention', odim=bidim)
            elif beamformer_type == 'mask_ss_attention_2':
                self.mask = MaskEstimator(btype, bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=1, cd_mask=False,
                                          SA=True, SA_type='lhuc_ss_attention_2', odim=bidim)
            elif beamformer_type == 'mask_angle_ss':
                self.mask = MaskEstimator(btype, ((2*n_channel)+3)*bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=1, cd_mask=False,
                                          SA=True, SA_type='lhuc_ss',
                                          odim=bidim, use_ipd=True)
            elif beamformer_type == 'mag_reg':
                self.mask = MagEstimator(btype, ((2*n_channel)+3)*bidim,
                                         blayers, bunits, bprojs,
                                         dropout_rate, nmask=1, odim=bidim,
                                         use_ipd=True)
            elif beamformer_type == 'mag_reg_xv':
                self.mask = MagEstimator(btype, bidim,
                                         blayers, bunits, bprojs,
                                         dropout_rate, nmask=1, cd_mask=False,
                                         SA=True, SA_type='sb', odim=bidim)
            elif beamformer_type == 'sb_mvdr_ad' or beamformer_type == 'sb_mvdr_pm_ad':
                self.mask = MaskEstimator(btype, 2*(n_channel+1)*bidim,
                                          blayers, bunits, bprojs,
                                          dropout_rate, nmask=3, cd_mask=True, odim=bidim)
                if beamformer_type == 'sb_mvdr_pm_ad':
                    self.post_mask = MaskEstimator(btype, bidim,
                                                   1, bunits, bprojs,
                                                   dropout_rate, nmask=1, post=True)
            if ref_channel == -1 and beamformer_type != 'lcmv' and beamformer_type != 'mask' and beamformer_type != 'mask_xv' and beamformer_type != 'mask_ss' and beamformer_type != 'mask_angle_ss':
                self.ref = AttentionReference(bidim, badim)
        else:
            self.mask = MaskEstimator(btype, bidim, blayers, bunits, bprojs,
                                      dropout_rate, nmask=2)
            self.ref = AttentionReference(bidim, badim)
        self.ref_channel = ref_channel
        self.beamformer_type = beamformer_type

    def forward(self, data, ilens, af=None, sv=None, xv=None, ss=None, ilens_ss=None, use_bf_test=False):
        """The forward function

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq

        Args:
            data (ComplexTensor): (B, T, C, F)
            ilens (torch.Tensor): (B,)
        Returns:
            enhanced (ComplexTensor): (B, T, F)
            ilens (torch.Tensor): (B,)

        """
        # data (B, T, C, F) -> (B, F, C, T)
        mask_interference = None
        mask_speech = None
        mask_noise = None
        mask_post = None
        sigma=None
        if data.dim() == 4:
            data = data.permute(0, 3, 2, 1)
        else:
            data = data.permute(0, 2, 1)
        #if ss is not None:
        #    ss = ss.permute(0, 2, 1)
        if self.beamformer_type == 'gdr' or self.beamformer_type == 'gdr_pm' or self.beamformer_type == 'sb_mvdr_pm'  or self.beamformer_type == 'sb_mvdr'  or self.beamformer_type == 'gdr_angle_ss':
            # mask: (B, F, C, T)
            data_1ch = (data[:,:,0,:].real ** 2 + data[:,:,0,:].imag ** 2) ** 0.5
            (mask_speech, mask_noise, mask_interference), _, se = self.mask(data_1ch, ilens, af, ss=ss, ilens_ss=ilens_ss)
            psd_s = get_power_spectral_density_matrix(data, mask_speech, averaging=False)
            psd_n = get_power_spectral_density_matrix(data, mask_noise, averaging=False)
            psd_i = get_power_spectral_density_matrix(data, mask_interference, averaging=False)
            if self.ref_channel < 0:
                u, _ = self.ref(psd_s, ilens)
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)),
                                device=data.device)
                u[..., self.ref_channel].fill_(1)
            if self.beamformer_type == 'gdr' or self.beamformer_type == 'gdr_pm' or self.beamformer_type == 'gdr_angle_ss':
                u2 = torch.zeros(*(data.size()[:-3] + (2,)),
                                device=data.device)
                u2[..., 0].fill_(1)
                sigma, _ = self.sdw(psd_s, psd_n, psd_i, ilens)
                ws = get_gdr_vector(psd_s, psd_n, sv, u, sigma, u2)
            else:
                ws = get_smvdr_vector(psd_s, psd_n, psd_i, u)
        elif self.beamformer_type == 'gdr2' or self.beamformer_type == 'gdr3':
            # mask: (B, F, C, T)
            data_1ch = (data[:,:,0,:].real ** 2 + data[:,:,0,:].imag ** 2) ** 0.5
            (mask_speech, mask_noise), _, se = self.mask(data_1ch, ilens, af, ss=ss, ilens_ss=ilens_ss)
            psd_s = get_power_spectral_density_matrix(data, mask_speech, averaging=False)
            psd_n = get_power_spectral_density_matrix(data, mask_noise, averaging=False)
            if self.ref_channel < 0:
                u, _ = self.ref(psd_s, ilens)
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)),
                                device=data.device)
                u[..., self.ref_channel].fill_(1)
            u2 = torch.zeros(*(data.size()[:-3] + (2,)),
                            device=data.device)
            u2[..., 0].fill_(1)
            sigma, _ = self.sdw(psd_s, psd_n, ilens)
            ws = get_gdr_vector(psd_s, psd_n, sv, u, sigma, u2)
        elif self.beamformer_type == 'mvdr_xv_sb':
            # mask: (B, F, C, T)
            (mask_speech, mask_noise, mask_interference), _, se = self.mask(data,
                                                                        ilens,
                                                                        af, xv)
            psd_s = get_power_spectral_density_matrix(data, mask_speech)
            psd_n = get_power_spectral_density_matrix(data, mask_noise)
            psd_i = get_power_spectral_density_matrix(data, mask_interference)
            if self.ref_channel < 0:
                u, _ = self.ref(psd_s, ilens)
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)),
                                device=data.device)
                u[..., self.ref_channel].fill_(1)
            ws = get_smvdr_vector(psd_s, psd_n, psd_i, u)
        elif self.beamformer_type == 'lcmv':
            # mask: (B, F, C, T)
            #(mask_noise,), _ = self.mask(data, ilens, None, None)
            #psd_n = get_power_spectral_density_matrix(data, mask_noise)
            data_1ch = (data[:,:,0,:].real ** 2 + data[:,:,0,:].imag ** 2) ** 0.5
            (mask_noise,), _, se = self.mask(data_1ch, ilens, af, None)
            psd_n = get_power_spectral_density_matrix(data, mask_noise, averaging=False)
            u = torch.zeros(*(data.size()[:-3] + (2,)),
                            device=data.device)
            u2 = torch.zeros(*(data.size()[:-3] + (2,)),
                             device=data.device)
            u[..., 0].fill_(1)
            u2[..., 1].fill_(1)
            ws, ws2 = get_lcmv_vector(psd_n, sv, u, u2)
        elif self.beamformer_type == 'mask':
            # mask: (B, F, C, T)
            data_1ch = (data[:,:,0,:].real ** 2 + data[:,:,0,:].imag ** 2) ** 0.5
            #data_1ch_l = (data_1ch + 1e-1).log()
            (mask_speech,), _, se = self.mask(data_1ch, ilens, af, None)
            if use_bf_test:
                #enhanced = (enhanced + 1e-1).log()
                mask_noise = torch.ones_like(mask_speech) - mask_speech
                psd_s = get_power_spectral_density_matrix(data, mask_speech,
                                                          averaging=False)
                psd_n = get_power_spectral_density_matrix(data, mask_noise,
                                                          averaging=False)
                u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)),
                                device=data.device)
                u[..., 0].fill_(1)
                ws = get_mvdr_vector(psd_s, psd_n, u)
                enhanced = apply_beamforming_vector(ws, data)
            else:
                enhanced = data_1ch * mask_speech

        elif self.beamformer_type == 'mask_asr':
            data_1ch = (data[:,:,0,:].real ** 2 + data[:,:,0,:].imag ** 2) ** 0.5
            #data_1ch_l = (data_1ch + 1e-1).log()
            (mask_speech,), _, se = self.mask(data_1ch, ilens, af, None)
            #enhanced = data_1ch_l * mask_speech
            #enhanced = enhanced.exp()
            mask_noise = torch.ones_like(mask_speech) - mask_speech
            psd_s = get_power_spectral_density_matrix(data, mask_speech,
                                                      averaging=False)
            psd_n = get_power_spectral_density_matrix(data, mask_noise,
                                                      averaging=False)
            if self.ref_channel < 0:
                u, _ = self.ref(psd_s, ilens)
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)),
                                device=data.device)
                u[..., self.ref_channel].fill_(1)
            ws = get_mvdr_vector(psd_s, psd_n, u)
            enhanced = apply_beamforming_vector(ws, data)
        elif self.beamformer_type == 'mvdr_angle_mimo':
            data_1ch = (data[:,:,0,:].real ** 2 + data[:,:,0,:].imag ** 2) ** 0.5
            #data_1ch_l = (data_1ch + 1e-1).log()
            masks, _, se = self.mask(data_1ch, ilens, af, None)
            mask_speech_list = list(masks[:-1])
            mask_noise = masks[-1]
            mask_speech = mask_noise
            psd_speeches = [get_power_spectral_density_matrix(data, mask, averaging=False) for
                            mask in mask_speech_list]
            psd_n = get_power_spectral_density_matrix(data, mask_noise,
                                                      averaging=False)
            enhanced = []
            ws = []
            for i in range(2):
                psd_s = psd_speeches.pop(i)
                if self.ref_channel < 0:
                    u, _ = self.ref(psd_s, ilens)
                else:
                    # (optional) Create onehot vector for fixed reference microphone
                    u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)),
                                    device=data.device)
                    u[..., self.ref_channel].fill_(1)
                w = get_mvdr_vector(psd_s, sum(psd_speeches) + psd_n, u)
                enh = apply_beamforming_vector(w, data)
                psd_speeches.insert(i, psd_s)
                enh = enh.transpose(-1, -2)
                enhanced.append(enh)
                ws.append(w)
        elif self.beamformer_type == 'mvdr_mimo':
            masks, _, se = self.mask(data, ilens)
            mask_speech_list = list(masks[:-1])
            mask_noise = masks[-1]
            mask_speech = mask_noise
            psd_speeches = [get_power_spectral_density_matrix(data, mask) for
                            mask in mask_speech_list]
            psd_n = get_power_spectral_density_matrix(data, mask_noise)
            enhanced = []
            ws = []
            for i in range(2):
                psd_s = psd_speeches.pop(i)
                if self.ref_channel < 0:
                    u, _ = self.ref(psd_s, ilens)
                else:
                    # (optional) Create onehot vector for fixed reference microphone
                    u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)),
                                    device=data.device)
                    u[..., self.ref_channel].fill_(1)
                w = get_mvdr_vector(psd_s, sum(psd_speeches) + psd_n, u)
                enh = apply_beamforming_vector(w, data)
                psd_speeches.insert(i, psd_s)
                enh = enh.transpose(-1, -2)
                enhanced.append(enh)
                ws.append(w)
        elif self.beamformer_type == 'mvdr_angle' or  self.beamformer_type =='mvdr_angle_ss' or self.beamformer_type == 'mvdr_angle_post' or self.beamformer_type == 'mvdr_angle_ss_pm' or self.beamformer_type == 'mvdr_angle_ss_post':
            data_1ch = (data[:,:,0,:].real ** 2 + data[:,:,0,:].imag ** 2) ** 0.5
            (mask_speech, mask_noise,), _, se = self.mask(data_1ch, ilens, af,
                                                          ss=ss,
                                                          ilens_ss=ilens_ss)
            psd_s = get_power_spectral_density_matrix(data, mask_speech,
                                                      averaging=False)
            psd_n = get_power_spectral_density_matrix(data, mask_noise,
                                                      averaging=False)
            if self.ref_channel < 0:
                u, _ = self.ref(psd_s, ilens)
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)),
                                device=data.device)
                u[..., self.ref_channel].fill_(1)
            ws = get_mvdr_vector(psd_s, psd_n, u)
        elif self.beamformer_type == 'mask_xv':
            # mask: (B, F, C, T)
            data_1ch = (data[:,:,0,:].real ** 2 + data[:,:,0,:].imag ** 2) ** 0.5
            #data.unsqueeze_(-1)
            #data = data.permute(0, 1, 3, 2)
            (mask_speech,), _, se = self.mask(data_1ch, ilens, None, xv)
            enhanced = data_1ch * mask_speech
            #enhanced = data[:,:,0,:] * mask_speech[:,:,0,:]
        elif self.beamformer_type == 'mask_angle_ss':
            # mask: (B, F, C, T)
            data_1ch = (data[:,:,0,:].real ** 2 + data[:,:,0,:].imag ** 2) ** 0.5
            (mask_speech,), _, se = self.mask(data_1ch, ilens, af, None, ss,
                                          ilens_ss)
            if use_bf_test:
                #enhanced = (enhanced + 1e-1).log()
                mask_noise = torch.ones_like(mask_speech) - mask_speech
                psd_s = get_power_spectral_density_matrix(data, mask_speech,
                                                          averaging=False)
                psd_n = get_power_spectral_density_matrix(data, mask_noise,
                                                          averaging=False)
                u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)),
                                device=data.device)
                u[..., 0].fill_(1)
                ws = get_mvdr_vector(psd_s, psd_n, u)
                enhanced = apply_beamforming_vector(ws, data)
            else:
                enhanced = data_1ch * mask_speech
        elif self.beamformer_type == 'mask_ss' or 'mask_ss_attention' in self.beamformer_type or self.beamformer_type =='mask_sb_ss':
            # mask: (B, F, C, T)
            data_1ch = (data[:,:,0,:].real ** 2 + data[:,:,0,:].imag ** 2) ** 0.5
            (mask_speech,), _, se = self.mask(data_1ch, ilens, None, None, ss,
                                          ilens_ss)
            if use_bf_test:
                #enhanced = (enhanced + 1e-1).log()
                mask_noise = torch.ones_like(mask_speech) - mask_speech
                psd_s = get_power_spectral_density_matrix(data, mask_speech,
                                                          averaging=False)
                psd_n = get_power_spectral_density_matrix(data, mask_noise,
                                                          averaging=False)
                u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)),
                                device=data.device)
                u[..., 0].fill_(1)
                ws = get_mvdr_vector(psd_s, psd_n, u)
                enhanced = apply_beamforming_vector(ws, data)
            else:
                enhanced = data_1ch * mask_speech
        elif self.beamformer_type == 'mvdr_ss':
            # mask: (B, F, C, T)
            (mask_speech, mask_noise), _, se = self.mask(data, ilens, None, None, ss,
                                          ilens_ss)
            #mask_noise = torch.ones_like(mask_speech) - mask_speech
            psd_s = get_power_spectral_density_matrix(data, mask_speech)
            psd_n = get_power_spectral_density_matrix(data, mask_noise)
            if self.ref_channel < 0:
                u, _ = self.ref(psd_s, ilens)
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)),
                                device=data.device)
                u[..., self.ref_channel].fill_(1)
            ws = get_mvdr_vector(psd_s, psd_n, u)
        elif self.beamformer_type == 'mag_reg':
            # mask: (B, F, C, T)
            mag = (data[:,:,0,:].real ** 2 + data[:,:,0,:].imag ** 2) ** 0.5
            (enhanced,), _, se = self.mask(mag, ilens, af, None)
        elif self.beamformer_type == 'mag_reg_xv':
            # mask: (B, F, C, T)
            data_1ch = (data[:,:,0,:].real ** 2 + data[:,:,0,:].imag ** 2) ** 0.5
            data_1ch.unsqueeze_(-1)
            data_1ch = data_1ch.permute(0, 1, 3, 2)
            (enhanced_mc,), _, se = self.mask(data_1ch, ilens, None, xv)
            enhanced = enhanced_mc[:,:,0,:]
        else:
            # mask: (B, F, C, T)
            (mask_speech, mask_noise), _, se = self.mask(data, ilens, af, xv)
            psd_speech = get_power_spectral_density_matrix(data, mask_speech)
            psd_noise = get_power_spectral_density_matrix(data, mask_noise)

            # u: (B, C)
            if self.ref_channel < 0:
                u, _ = self.ref(psd_speech, ilens)
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)),
                                device=data.device)
                u[..., self.ref_channel].fill_(1)

            ws = get_mvdr_vector(psd_speech, psd_noise, u)

        if 'mimo' in self.beamformer_type:
            if 'angle' in self.beamformer_type:
                mask_speech = mask_speech.transpose(-1, -2)
                mask_noise = mask_noise.transpose(-1, -2)
            else:
                mask_speech = mask_speech.transpose(-1, -3)
                mask_noise = mask_noise.transpose(-1, -3)
        else:
            if 'mask' not in self.beamformer_type:
                enhanced = apply_beamforming_vector(ws, data)
                if self.beamformer_type == 'lcmv':
                    enhanced_2 = apply_beamforming_vector(ws2, data)
                    enhanced_2 = (enhanced_2.real ** 2 + enhanced_2.imag ** 2) ** 0.5
                    mask_interference = enhanced_2.transpose(-1, -2)

            if 'pm' not in self.beamformer_type:
                # (..., F, T) -> (..., T, F)
                enhanced = enhanced.transpose(-1, -2)
            if mask_speech is not None:
                if self.beamformer_type == 'mvdr' or  self.beamformer_type == 'mvdr_ss':
                    mask_speech = mask_speech.transpose(-1, -3)
                else:
                    mask_speech = mask_speech.transpose(-1, -2)
            else:
                mask_speech = mask_noise
                mask_speech = mask_speech.transpose(-1, -2)
            if mask_noise is not None:
                if self.beamformer_type == 'mvdr' or  self.beamformer_type == 'mvdr_ss':
                    mask_noise = mask_noise.transpose(-1, -3)
                else:
                    mask_noise = mask_noise.transpose(-1, -2)
            if mask_interference is not None:
                if self.beamformer_type == 'mvdr' or  self.beamformer_type == 'mvdr_ss':
                    mask_interference = mask_interference.transpose(-1, -3)
                else:
                    mask_interference = mask_interference.transpose(-1, -2)
            if 'pm' in self.beamformer_type:
                (mask_post, ), _, se = self.post_mask(enhanced, ilens, None, None, ss,
                                                      ilens_ss)
                mask_post = mask_post.permute(0, 2, 1)
                enhanced = enhanced.transpose(-1, -2)
                enhanced = enhanced.real ** 2 + enhanced.imag ** 2
                enhanced = enhanced * mask_post
            if 'post' in self.beamformer_type:
                enhanced = enhanced.real ** 2 + enhanced.imag ** 2
                enhanced = enhanced * mask_speech

        return enhanced, ilens, mask_speech, mask_noise, mask_interference, mask_post, sigma, se


class AttentionReference(torch.nn.Module):
    def __init__(self, bidim, att_dim):
        super().__init__()
        self.mlp_psd = torch.nn.Linear(bidim, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)

    def forward(self, psd_in: ComplexTensor, ilens: torch.LongTensor,
                scaling: float = 2.0) -> Tuple[torch.Tensor, torch.LongTensor]:
        """The forward function

        Args:
            psd_in (ComplexTensor): (B, F, C, C)
            ilens (torch.Tensor): (B,)
            scaling (float):
        Returns:
            u (torch.Tensor): (B, C)
            ilens (torch.Tensor): (B,)
        """
        B, _, C = psd_in.size()[:3]
        assert psd_in.size(2) == psd_in.size(3), psd_in.size()
        # psd_in: (B, F, C, C)
        psd = psd_in.masked_fill(torch.eye(C, dtype=torch.uint8,
                                           device=psd_in.device), 0)
        # psd: (B, F, C, C) -> (B, C, F)
        psd = (psd.sum(dim=-1) / (C - 1)).transpose(-1, -2)

        # Calculate amplitude
        psd_feat = (psd.real ** 2 + psd.imag ** 2) ** 0.5

        # (B, C, F) -> (B, C, F2)
        mlp_psd = self.mlp_psd(psd_feat)
        # (B, C, F2) -> (B, C, 1) -> (B, C)
        e = self.gvec(torch.tanh(mlp_psd)).squeeze(-1)
        u = F.softmax(scaling * e, dim=-1)
        return u, ilens


class SDW_net(torch.nn.Module):
    def __init__(self, bidim):
        super().__init__()
        self.mlp_psd = torch.nn.Linear(3*bidim, bidim)

    def forward(self, psd_s: ComplexTensor,
                psd_n: ComplexTensor,
                psd_i: ComplexTensor,
                ilens: torch.LongTensor,
                scaling: float = 2.0) -> Tuple[torch.Tensor, torch.LongTensor]:
        """The forward function

        Args:
            psd_in (ComplexTensor): (B, F, C, C)
            ilens (torch.Tensor): (B,)
            scaling (float):
        Returns:
            u (torch.Tensor): (B, C)
            ilens (torch.Tensor): (B,)
        """
        B, _, C = psd_s.size()[:3]
        assert psd_s.size(2) == psd_s.size(3), psd_s.size()
        B, _, C = psd_n.size()[:3]
        assert psd_n.size(2) == psd_n.size(3), psd_n.size()
        B, _, C = psd_i.size()[:3]
        assert psd_i.size(2) == psd_i.size(3), psd_i.size()
        # psd_s: (B, F, C, C)
        psd_s_pad = psd_s.masked_fill(torch.eye(C, dtype=torch.uint8,
                                                device=psd_s.device), 0)
        # psd_s_pad: (B, F, C, C) -> (B, F)
        psd_s_pad = psd_s_pad.sum(dim=(-1, -2)) / ((C - 1)*(C - 1))
        psd_s_feat = (psd_s_pad.real ** 2 + psd_s_pad.imag ** 2) ** 0.5
        psd_n_pad = psd_n.masked_fill(torch.eye(C, dtype=torch.uint8,
                                                device=psd_n.device), 0)
        psd_n_pad = psd_n_pad.sum(dim=(-1, -2)) / ((C - 1)*(C - 1))
        psd_n_feat = (psd_n_pad.real ** 2 + psd_n_pad.imag ** 2) ** 0.5
        psd_i_pad = psd_i.masked_fill(torch.eye(C, dtype=torch.uint8,
                                                device=psd_i.device), 0)
        psd_i_pad = psd_i_pad.sum(dim=(-1, -2)) / ((C - 1)*(C - 1))
        psd_i_feat = (psd_i_pad.real ** 2 + psd_i_pad.imag ** 2) ** 0.5
        psd_feat = torch.cat((psd_s_feat, psd_n_feat, psd_i_feat), -1)

        mlp_psd = self.mlp_psd(psd_feat)
        u = torch.clamp(mlp_psd, min=0, max=1)
        return u, ilens


class SDW_net_2(torch.nn.Module):
    def __init__(self, bidim, two_dim=True):
        super().__init__()
        if two_dim:
            self.mlp_psd = torch.nn.Linear(2*bidim, bidim)
        else:
            self.mlp_psd = torch.nn.Linear(2*bidim, 1)

    def forward(self, psd_s: ComplexTensor,
                psd_n: ComplexTensor,
                ilens: torch.LongTensor,
                scaling: float = 2.0) -> Tuple[torch.Tensor, torch.LongTensor]:
        """The forward function

        Args:
            psd_in (ComplexTensor): (B, F, C, C)
            ilens (torch.Tensor): (B,)
            scaling (float):
        Returns:
            u (torch.Tensor): (B, C)
            ilens (torch.Tensor): (B,)
        """
        B, _, C = psd_s.size()[:3]
        assert psd_s.size(2) == psd_s.size(3), psd_s.size()
        B, _, C = psd_n.size()[:3]
        assert psd_n.size(2) == psd_n.size(3), psd_n.size()
        # psd_s: (B, F, C, C)
        psd_s_pad = psd_s.masked_fill(torch.eye(C, dtype=torch.uint8,
                                                device=psd_s.device), 0)
        # psd_s_pad: (B, F, C, C) -> (B, F)
        psd_s_pad = psd_s_pad.sum(dim=(-1, -2)) / ((C - 1)*(C - 1))
        psd_s_feat = (psd_s_pad.real ** 2 + psd_s_pad.imag ** 2) ** 0.5
        psd_n_pad = psd_n.masked_fill(torch.eye(C, dtype=torch.uint8,
                                                device=psd_n.device), 0)
        psd_n_pad = psd_n_pad.sum(dim=(-1, -2)) / ((C - 1)*(C - 1))
        psd_n_feat = (psd_n_pad.real ** 2 + psd_n_pad.imag ** 2) ** 0.5
        psd_feat = torch.cat((psd_s_feat, psd_n_feat), -1)

        mlp_psd = self.mlp_psd(psd_feat)
        u = torch.clamp(mlp_psd, min=0, max=1)
        return u, ilens
