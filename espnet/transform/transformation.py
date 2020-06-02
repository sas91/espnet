from collections import OrderedDict
import copy
import io
import json
import logging
import sys
import pudb

from espnet.utils.dynamic_import import dynamic_import

PY2 = sys.version_info[0] == 2

if PY2:
    from collections import Sequence
    from funcsigs import signature
else:
    # The ABCs from 'collections' will stop working in 3.8
    from collections.abc import Sequence
    from inspect import signature


import_alias = dict(
    identity="espnet.transform.transform_interface:Identity",
    time_warp="espnet.transform.spec_augment:TimeWarp",
    time_mask="espnet.transform.spec_augment:TimeMask",
    freq_mask="espnet.transform.spec_augment:FreqMask",
    spec_augment="espnet.transform.spec_augment:SpecAugment",
    speed_perturbation='espnet.transform.perturb:SpeedPerturbation',
    volume_perturbation='espnet.transform.perturb:VolumePerturbation',
    noise_injection='espnet.transform.perturb:NoiseInjection',
    bandpass_perturbation='espnet.transform.perturb:BandpassPerturbation',
    rir_convolve='espnet.transform.perturb:RIRConvolve',
    delta='espnet.transform.add_deltas:AddDeltas',
    cmvn='espnet.transform.cmvn:CMVN',
    utterance_cmvn='espnet.transform.cmvn:UtteranceCMVN',
    fbank='espnet.transform.spectrogram:LogMelSpectrogram',
    spectrogram='espnet.transform.spectrogram:Spectrogram',
    stft='espnet.transform.spectrogram:Stft',
    stft_pit='espnet.transform.spectrogram:StftPIT',
    stft_stv_af='espnet.transform.spectrogram:StftAf',
    mag_af='espnet.transform.spectrogram:MagAf',
    stft_phase='espnet.transform.spectrogram:Phase',
    stft_phase_doa='espnet.transform.spectrogram:PhaseDOA',
    stft_ipd='espnet.transform.spectrogram:IPD',
    istft='espnet.transform.spectrogram:IStft',
    stft2fbank='espnet.transform.spectrogram:Stft2LogMelSpectrogram',
    wpe='espnet.transform.wpe:WPE',
    channel_selector='espnet.transform.channel_selector:ChannelSelector')


class Transformation(object):
    """Apply some functions to the mini-batch

    Examples:
        >>> kwargs = {"process": [{"type": "fbank",
        ...                        "n_mels": 80,
        ...                        "fs": 16000},
        ...                       {"type": "cmvn",
        ...                        "stats": "data/train/cmvn.ark",
        ...                        "norm_vars": True},
        ...                       {"type": "delta", "window": 2, "order": 2}]}
        >>> transform = Transformation(kwargs)
        >>> bs = 10
        >>> xs = [np.random.randn(100, 80).astype(np.float32)
        ...       for _ in range(bs)]
        >>> xs = transform(xs)
    """

    def __init__(self, conffile=None):
        #pudb.set_trace()
        if conffile is not None:
            if isinstance(conffile, dict):
                self.conf = copy.deepcopy(conffile)
            else:
                with io.open(conffile, encoding='utf-8') as f:
                    self.conf = json.load(f)
                    assert isinstance(self.conf, dict), type(self.conf)
        else:
            self.conf = {'mode': 'sequential', 'process': []}

        self.functions = OrderedDict()
        if self.conf.get('mode', 'sequential') == 'sequential':
            for idx, process in enumerate(self.conf['process']):
                assert isinstance(process, dict), type(process)
                opts = dict(process)
                process_type = opts.pop('type')
                class_obj = dynamic_import(process_type, import_alias)
                try:
                    self.functions[idx] = class_obj(**opts)
                except TypeError:
                    try:
                        signa = signature(class_obj)
                    except ValueError:
                        # Some function, e.g. built-in function, are failed
                        pass
                    else:
                        logging.error('Expected signature: {}({})'
                                      .format(class_obj.__name__, signa))
                    raise
        else:
            raise NotImplementedError(
                'Not supporting mode={}'.format(self.conf['mode']))

    def __repr__(self):
        rep = '\n' + '\n'.join(
            '    {}: {}'.format(k, v) for k, v in self.functions.items())
        return '{}({})'.format(self.__class__.__name__, rep)

    def __call__(self, xs, uttid_list=None, angle1_list=None, angle2_list=None,
                 xvector_list=None, clean_list=None, ss_list=None,
                 ss_list_2=None, ss_list_3=None, s1_list=None, s2_list=None,
                 s3_list=None, s4_list=None, mix_list=None, noise_list=None, **kwargs):
        """Return new mini-batch

        :param Union[Sequence[np.ndarray], np.ndarray] xs:
        :param Union[Sequence[str], str] uttid_list:
        :return: batch:
        :rtype: List[np.ndarray]
        """
        #pudb.set_trace()
        if not isinstance(xs, Sequence):
            is_batch = False
            xs = [xs]
        else:
            is_batch = True

        if isinstance(uttid_list, str):
            uttid_list = [uttid_list for _ in range(len(xs))]

        if self.conf.get('mode', 'sequential') == 'sequential':
            for idx in range(len(self.conf['process'])):
                func = self.functions[idx]

                # Derive only the args which the func has
                try:
                    param = signature(func).parameters
                except ValueError:
                    # Some function, e.g. built-in function, are failed
                    param = {}
                _kwargs = {k: v for k, v in kwargs.items()
                           if k in param}
                try:
                    if uttid_list is not None and 'uttid' in param:
                        xs = [func(x, u, **_kwargs)
                              for x, u in zip(xs, uttid_list)]
                    else:
                        if angle1_list and angle2_list and xvector_list and clean_list and ss_list and ss_list_2 and ss_list_3:
                            xs = [func(x, a1, a2, xv, c, ss, ss2, ss3, **_kwargs)
                                  for x, a1, a2, xv, c, ss, ss2, ss3 in zip(xs, angle1_list,
                                                                            angle2_list,
                                                                            xvector_list,
                                                                            clean_list,
                                                                            ss_list,
                                                                            ss_list_2,
                                                                            ss_list_3)]
                        elif s1_list and s2_list and s3_list and s4_list and noise_list:
                            xs = [func(x, s1, s2, s3, s4, n=n, **_kwargs)
                                  for x, s1, s2, s3, s4, n in zip(xs, s1_list,
                                                                            s2_list,
                                                                            s3_list,
                                                                            s4_list,
                                                                            noise_list)]
                        elif mix_list and noise_list:
                            xs = [func(x, m=m, n=n, **_kwargs)
                                  for x, m, n in zip(xs, mix_list,
                                                      noise_list)]
                        elif s1_list and noise_list and xvector_list:
                            xs = [func(x, s1=s1, n=n, xv=xv, **_kwargs)
                                  for x, s1, n, xv in zip(xs, s1_list,
                                                          noise_list,
                                                          xvector_list)]
                        elif s1_list and noise_list and ss_list and ss_list_2 and ss_list_3:
                            xs = [func(x, s1=s1, n=n, ss=ss, ss2=ss2, ss3=ss3, **_kwargs)
                                  for x, s1, n, ss, ss2, ss3 in zip(xs, s1_list,
                                                                        noise_list,
                                                                        ss_list,
                                                                        ss_list_2,
                                                                        ss_list_3)]
                        elif angle1_list and angle2_list and clean_list and ss_list and ss_list_2 and ss_list_3:
                            xs = [func(x, a1, a2, c=c, ss=ss, ss2=ss2, ss3=ss3, **_kwargs)
                                  for x, a1, a2, c, ss, ss2, ss3 in zip(xs, angle1_list,
                                                                            angle2_list,
                                                                            clean_list,
                                                                            ss_list,
                                                                            ss_list_2,
                                                                            ss_list_3)]
                        elif angle1_list and angle2_list and ss_list and ss_list_2 and ss_list_3:
                            xs = [func(x, a1, a2, ss=ss, ss2=ss2, ss3=ss3, **_kwargs)
                                  for x, a1, a2, ss, ss2, ss3 in zip(xs, angle1_list,
                                                                         angle2_list,
                                                                         ss_list,
                                                                         ss_list_2,
                                                                         ss_list_3)]
                        elif angle1_list and angle2_list and xvector_list and clean_list and ss_list:
                            xs = [func(x, a1, a2, xv, c, ss, **_kwargs)
                                  for x, a1, a2, xv, c, ss in zip(xs, angle1_list,
                                                                  angle2_list,
                                                                  xvector_list,
                                                                  clean_list,
                                                                  ss_list)]
                        elif angle1_list and angle2_list and clean_list:
                            xs = [func(x, a1, a2, c=c, **_kwargs)
                                  for x, a1, a2, c in zip(xs, angle1_list, angle2_list, clean_list)]
                        elif xvector_list and clean_list:
                            xs = [func(x, xv=xv, c=c, **_kwargs)
                                  for x, xv, c in zip(xs, xvector_list, clean_list)]
                        elif ss_list and ss_list_2 and ss_list_3 and clean_list:
                            xs = [func(x, ss=ss, ss2=ss2, ss3=ss3, c=c, **_kwargs)
                                  for x, ss, ss2, ss3, c in zip(xs, ss_list,
                                                                ss_list_2,
                                                                ss_list_3,
                                                                clean_list)]
                        elif angle1_list and angle2_list and xvector_list:
                            xs = [func(x, a1, a2, xv, **_kwargs)
                                  for x, a1, a2, xv in zip(xs, angle1_list,
                                                           angle2_list,
                                                           xvector_list)]
                        elif angle1_list and angle2_list:
                            xs = [func(x, a1, a2, **_kwargs)
                                  for x, a1, a2 in zip(xs, angle1_list, angle2_list)]
                        elif xvector_list:
                            xs = [func(x, xv=xv, **_kwargs)
                                  for x, xv in zip(xs, xvector_list)]
                        elif ss_list and ss_list_2 and ss_list_3:
                            xs = [func(x, ss=ss, ss2=ss2, ss3=ss3, **_kwargs)
                                  for x, ss, ss2, ss3 in zip(xs, ss_list,
                                                             ss_list_2, ss_list_3)]
                        else:
                            xs = [func(x, **_kwargs) for x in xs]
                except Exception:
                    logging.fatal('Catch a exception from {}th func: {}'
                                  .format(idx, func))
                    raise
        else:
            raise NotImplementedError(
                'Not supporting mode={}'.format(self.conf['mode']))

        if is_batch:
            return xs
        else:
            return xs[0]
