#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import copy
import json
import logging
import math
import os
import sys
import pudb
import pickle
import librosa

from chainer.datasets import TransformDataset
from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions
import numpy as np
from tensorboardX import SummaryWriter
import torch

from espnet.enhance.asr_utils import adadelta_eps_decay
from espnet.enhance.asr_utils import add_results_to_json
from espnet.enhance.asr_utils import CompareValueTrigger
from espnet.enhance.asr_utils import get_model_conf
from espnet.enhance.asr_utils import make_batchset
from espnet.enhance.asr_utils import plot_spectrogram
from espnet.enhance.asr_utils import restore_snapshot
from espnet.enhance.asr_utils import torch_load
from espnet.enhance.asr_utils import torch_resume
from espnet.enhance.asr_utils import torch_save
from espnet.enhance.asr_utils import torch_snapshot
from espnet.nets.pytorch_backend.e2e_asr import pad_list
from espnet.transform.spectrogram import IStft
from espnet.transform.spectrogram import Stft
from espnet.transform.transformation import Transformation
from espnet.utils.cli_utils import FileWriterWrapper
from espnet.utils.gev import gev_wrapper_on_masks
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.iterators import ToggleableShufflingMultiprocessIterator
from espnet.utils.training.iterators import ToggleableShufflingSerialIterator
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

import matplotlib
matplotlib.use('Agg')

if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest as zip_longest

REPORT_INTERVAL = 5


class CustomEvaluator(extensions.Evaluator):
    """Custom Evaluator for Pytorch

    :param torch.nn.Module model : The model to evaluate
    :param chainer.dataset.Iterator : The train iterator
    :param target :
    :param CustomConverter converter : The batch converter
    :param torch.device device : The device used
    """

    def __init__(self, model, iterator, target, converter, device):
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.converter = converter
        self.device = device

    # The core part of the update routine can be customized by overriding
    def evaluate(self):
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                observation = {}
                with reporter_module.report_scope(observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    x = self.converter(batch, self.device)
                    self.model(*x)
                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class CustomUpdater(training.StandardUpdater):
    """Custom Updater for Pytorch

    :param torch.nn.Module model : The model to update
    :param int grad_clip_threshold : The gradient clipping value to use
    :param chainer.dataset.Iterator train_iter : The training iterator
    :param torch.optim.optimizer optimizer: The training optimizer
    :param CustomConverter converter: The batch converter
    :param torch.device device : The device to use
    :param int ngpu : The number of gpus to use
    """

    def __init__(self, model, grad_clip_threshold, train_iter,
                 optimizer, converter, device, ngpu, accum_grad=1):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        self.converter = converter
        self.device = device
        self.ngpu = ngpu
        self.accum_grad = accum_grad
        self.forward_count = 0

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch ( a list of json files)
        batch = train_iter.next()
        x = self.converter(batch, self.device)

        # Compute the loss at this time step and accumulate it
        loss = self.model(*x).mean() / self.accum_grad
        loss.backward()  # Backprop
        loss.detach()  # Truncate the graph

        # update parameters
        self.forward_count += 1
        if self.forward_count != self.accum_grad:
            return
        self.forward_count = 0
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()
        optimizer.zero_grad()


class CustomConverter(object):
    """Custom batch converter for Pytorch

    :param int subsampling_factor : The subsampling factor
    """

    def __init__(self, subsampling_factor=1, loss='mse', bftype='mask'):
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.loss = loss
        self.bftype = bftype

    def __call__(self, batch, device):
        """Transforms a batch and send it to a device

        :param list batch: The batch to transform
        :param torch.device device: The device to send to
        :return: a tuple xs_pad, ilens, ys_pad
        :rtype (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        # batch should be located in list
        assert len(batch) == 1
        xs = batch[0][0]
        xs_ft = [np.abs(x['stft']) for x in xs]
        if 'pit' in self.bftype:
            if self.loss == 'bce':
                xs_s1 = [x['m1'] for x in xs]
                xs_s2 = [x['m2'] for x in xs]
                xs_s3 = [x['m3'] for x in xs]
                xs_s4 = [x['m4'] for x in xs]
                xs_n = [x['mn'] for x in xs]
            else:
                xs_s1 = [x['s1'] for x in xs]
                xs_s2 = [x['s2'] for x in xs]
                xs_s3 = [x['s3'] for x in xs]
                xs_s4 = [x['s4'] for x in xs]
                xs_n = [x['n'] for x in xs]
        elif 'denoise' in self.bftype:
            xs_s1 = [x['m'] for x in xs]
            xs_n = [x['n'] for x in xs]
        elif 'xvec' in self.bftype:
            xs_s1 = [x['s1'] for x in xs]
            xs_n = [x['n'] for x in xs]
            xs_xv = [x['xv'] for x in xs]
        elif 'ts' in self.bftype:
            xs_s1 = [x['s1'] for x in xs]
            xs_n = [x['n'] for x in xs]
            xs_ss = [x['ss'] for x in xs]
            ilens_ss = np.array([x.shape[0] for x in xs_ss])


        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs_ft])

        # perform padding and convert to tensor
        xs_pad = pad_list(
            [torch.from_numpy(x).float() for x in xs_ft], 0).to(device)
        if 'pit' in self.bftype:
            xs_pad_s1 = pad_list(
                [torch.from_numpy(x).float() for x in xs_s1], -1).to(device)
            xs_pad_s2 = pad_list(
                [torch.from_numpy(x).float() for x in xs_s2], -1).to(device)
            xs_pad_s3 = pad_list(
                [torch.from_numpy(x).float() for x in xs_s3], -1).to(device)
            xs_pad_s4 = pad_list(
                [torch.from_numpy(x).float() for x in xs_s4], -1).to(device)
            xs_pad_n = pad_list(
                [torch.from_numpy(x).float() for x in xs_n], -1).to(device)
            ys_pad = {'s1': xs_pad_s1,
                      's2': xs_pad_s2,
                      's3': xs_pad_s3,
                      's4': xs_pad_s4,
                      'n': xs_pad_n}
        elif 'denoise' in self.bftype:
            xs_pad_s1 = pad_list(
                [torch.from_numpy(x).float() for x in xs_s1], -1).to(device)
            xs_pad_n = pad_list(
                [torch.from_numpy(x).float() for x in xs_n], -1).to(device)
            ys_pad = {'s1': xs_pad_s1,
                      'n': xs_pad_n}
        elif 'xvec' in self.bftype:
            xs_pad_s1 = pad_list(
                [torch.from_numpy(x).float() for x in xs_s1], -1).to(device)
            xs_pad_n = pad_list(
                [torch.from_numpy(x).float() for x in xs_n], -1).to(device)
            xs_pad_xv = pad_list(
                [torch.from_numpy(x).float() for x in xs_xv], 0).to(device)
            ys_pad = {'s1': xs_pad_s1,
                      'n': xs_pad_n,
                      'xv': xs_pad_xv}
        elif 'ts' in self.bftype:
            xs_pad_s1 = pad_list(
                [torch.from_numpy(x).float() for x in xs_s1], -1).to(device)
            xs_pad_n = pad_list(
                [torch.from_numpy(x).float() for x in xs_n], -1).to(device)
            xs_pad_ss = pad_list(
                [torch.from_numpy(x).float() for x in xs_ss], 0).to(device)
            ilens_ss = torch.from_numpy(ilens_ss).to(device)
            ys_pad = {'s1': xs_pad_s1,
                      'n': xs_pad_n,
                      'ilens_ss': ilens_ss,
                      'ss': xs_pad_ss}

        ilens = torch.from_numpy(ilens).to(device)

        return xs_pad, ilens, ys_pad


def train(args):
    """Train with the given args

    :param Namespace args: The program arguments
    """
    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][-1])
    logging.info('#input dims : ' + str(idim))

    # specify model architecture
    model_class = dynamic_import(args.model_module)
    model = model_class(idim, args)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, vars(args)),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.opt == 'noam':
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    converter = CustomConverter(bftype=args.bftype, loss=args.loss_type)

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_in, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1, shortest_first=use_sortagrad)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_in, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1)

    load_tr = LoadInputsAndTargets(
        mode='separation', bftype=args.bftype, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode='separation', bftype=args.bftype, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    if args.n_iter_processes > 0:
        train_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(train, load_tr),
            batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20,
            shuffle=not use_sortagrad)
        valid_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(valid, load_cv),
            batch_size=1, repeat=False, shuffle=False,
            n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
    else:
        train_iter = ToggleableShufflingSerialIterator(
            TransformDataset(train, load_tr),
            batch_size=1, shuffle=not use_sortagrad)
        valid_iter = ToggleableShufflingSerialIterator(
            TransformDataset(valid, load_cv),
            batch_size=1, repeat=False, shuffle=False)

    # Set up a trainer
    updater = CustomUpdater(
        model, args.grad_clip, train_iter, optimizer, converter, device, args.ngpu, args.accum_grad)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    if use_sortagrad:
        trainer.extend(ShufflingEnabler([train_iter]),
                       trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, 'epoch'))

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluator(model, valid_iter, reporter, converter, device))

    # Save attention weight each epoch

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                         'epoch', file_name='loss.png'))

    trainer.extend(extensions.snapshot_object(model, 'model.loss.best', savefun=torch_save),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))

    # save snapshot which contains model and optimizer states
    trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                       trigger=CompareValueTrigger(
                           'validation/main/loss',
                           lambda best_value, current_value: best_value < current_value))
        trainer.extend(adadelta_eps_decay(args.eps_decay),
                       trigger=CompareValueTrigger(
                           'validation/main/loss',
                           lambda best_value, current_value: best_value < current_value))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(REPORT_INTERVAL, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'elapsed_time']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
            trigger=(REPORT_INTERVAL, 'iteration'))
        report_keys.append('eps')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(REPORT_INTERVAL, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=REPORT_INTERVAL))
    set_early_stop(trainer, args)

    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        writer = SummaryWriter(log_dir=args.tensorboard_dir)
    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)


def enhance_chime6(args):
    """Dumping enhanced speech and mask

    :param Namespace args: The program arguments
    """
    set_deterministic_pytorch(args)
    # read training config
    idim, train_args = get_model_conf(args.model, args.model_conf)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, train_args)
    torch_load(args.model, model)
    model.recog_args = args

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='separation', bftype=train_args.bftype, load_output=False, sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False})

    #load_inputs_and_targets = LoadInputsAndTargets(
    #    mode='asr', load_output=False, sort_in_input_length=False,
    #    preprocess_conf=None  # Apply pre_process in outer func
    #)
    if args.batchsize == 0:
        args.batchsize = 1

    # Creates writers for outputs from the network
    if args.enh_wspecifier is not None:
        enh_writer = FileWriterWrapper(args.enh_wspecifier,
                                       filetype=args.enh_filetype)
    else:
        enh_writer = None

    # Creates a Transformation instance
    preprocess_conf = (
        train_args.preprocess_conf if args.preprocess_conf is None
        else args.preprocess_conf)
    #if preprocess_conf is not None:
    #    logging.info('Use preprocessing'.format(preprocess_conf))
    #    transform = Transformation(preprocess_conf)
    #else:
    #    transform = None

    # Creates a IStft instance
    istft = None
    stft = None
    frame_shift = args.istft_n_shift  # Used for plot the spectrogram
    if args.apply_istft:
        if preprocess_conf is not None:
            # Read the conffile and find stft setting
            with open(preprocess_conf) as f:
                # Json format: e.g.
                #    {"process": [{"type": "stft",
                #                  "win_length": 400,
                #                  "n_fft": 512, "n_shift": 160,
                #                  "window": "han"},
                #                 {"type": "foo", ...}, ...]}
                conf = json.load(f)
                assert 'process' in conf, conf
                # Find stft setting
                for p in conf['process']:
                    if p['type'] == 'stft_pit':
                        istft = IStft(win_length=p['win_length'],
                                      n_shift=p['n_shift'],
                                      window=p.get('window', 'hann'))
                        stft = Stft(win_length=p['win_length'],
                                    n_shift=p['n_shift'],
                                    n_fft=p['n_fft'],
                                    window=p.get('window', 'hann'))
                        logging.info('stft is found in {}. '
                                     'Setting istft config from it\n{}'
                                     .format(preprocess_conf, istft))
                        frame_shift = p['n_shift']
                        nfft = p['n_fft']
                        break
        if istft is None:
            # Set from command line arguments
            istft = IStft(win_length=args.istft_win_length,
                          n_shift=args.istft_n_shift,
                          window=args.istft_window)
            logging.info('Setting istft config from the command line args\n{}'
                         .format(istft))

    # sort data
    keys = list(js.keys())
    feat_lens = [js[key]['input'][0]['shape'][0] for key in keys]
    sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
    keys = [keys[i] for i in sorted_index]

    def grouper(n, iterable, fillvalue=None):
        kargs = [iter(iterable)] * n
        return zip_longest(*kargs, fillvalue=fillvalue)

    num_images = 0
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)

    for names in grouper(args.batchsize, keys, None):
        batch = [(name, js[name]) for name in names]

        enh_full = []
        enh_gev_full = []
        # May be in time region: (Batch, [Time, Channel])
        #feats_full = load_inputs_and_targets(batch)[0][0]['stft']
        #nc = load_inputs_and_targets(batch)[0][0]['nc']
        #tlen, clen, flen = feats_full.shape
        feats_full = load_inputs_and_targets(batch)[0][0]
        tlen, clen = feats_full.shape
        #org_feats = load_inputs_and_targets(batch)[0]
        #if transform is not None:
        #    # May be in time-freq region: : (Batch, [Time, Channel, Freq])
        #    feats = transform(org_feats, train=False)
        #else:
        #    feats = org_feats

        for i in range(0,args.segments):
            begin=i*int(tlen/args.segments)
            end=(i+1)*int(tlen/args.segments)
            end+=frame_shift
            if end > tlen:
                end = tlen
                offset = 0
            else:
                offset = frame_shift
            #feats = feats_full[begin:end,:,:]
            feats = feats_full[begin:end,:]
            length = len(feats) - offset
            #feats_pad = librosa.util.fix_length(feats, length + nfft // 2)
            nc=np.max(np.abs(feats))
            if nc > 0.0:
                feats=feats/nc
            feats = stft(feats)
            feats = feats[None, :]
            with torch.no_grad():
                ms1, mn, ms2, ms3, ms4, enh, ilens = model.enhance(feats, use_bf_test=True)
            enh_gev=gev_wrapper_on_masks(feats[0,:], mn[0,:], ms1[0,:])
            # Plot spectrogram
            if args.image_dir is not None and num_images < args.num_images:
                import matplotlib.pyplot as plt
                plt.rcParams.update({'font.size': 20})
                plt.tight_layout()
                num_images += 1
                plt.clf()
                plt.figure(figsize=(20, 10))
                plt.subplot(5, 1, 1)
                plt.title('Mixture')
                plot_spectrogram(plt, np.abs(feats[0,:,0,:]).T, fs=args.fs,
                                 mode='db', frame_shift=frame_shift,
                                 bottom=False, labelbottom=False)

                plt.subplot(5, 1, 2)
                plt.title('Speech Mask')
                plot_spectrogram(plt, ms1[0,:,:,0], fs=args.fs, mode='linear', frame_shift=frame_shift, bottom=False, labelbottom=False)

                plt.subplot(5, 1, 3)
                plt.title('Noise Mask')
                plot_spectrogram(plt, mn[0,:,:,0], fs=args.fs, mode='linear', frame_shift=frame_shift, bottom=False, labelbottom=False)

                plt.subplot(5, 1, 4)
                plt.title('Enhanced - MVDR')
                plot_spectrogram(plt, np.abs(enh[0,:,:]).T, fs=args.fs,
                                 mode='db', frame_shift=frame_shift)

                plt.subplot(5, 1, 5)
                plt.title('Enhanced - GEV')
                plot_spectrogram(plt, np.abs(enh_gev).T, fs=args.fs,
                                 mode='db', frame_shift=frame_shift)
                plt.savefig(os.path.join(args.image_dir, 'a' + str(num_images) + '.png'))
            enh = istft(enh[0,:, ], length=length)
            enh = nc * enh
            enh_full.append(enh)
            enh_gev = istft(enh_gev, length=length)
            enh_gev = nc * enh_gev
            enh_gev_full.append(enh_gev)

        if tlen -257 > args.segments*int(tlen/args.segments):
            begin=args.segments*int(tlen/args.segments)
            #feats = feats_full[begin:,:,:]
            feats = feats_full[begin:,:]
            length = len(feats)
            nc=np.max(np.abs(feats))
            if nc > 0.001:
                feats=feats/nc
            feats = stft(feats)
            feats = feats[None, :]
            with torch.no_grad():
                ms1, mn, ms2, ms3, ms4, enh, ilens = model.enhance(feats, use_bf_test=True)
            enh_gev=gev_wrapper_on_masks(feats[0,:], mn[0,:], ms1[0,:])
            enh = istft(enh[0,:,:], length=length)
            enh = nc * enh
            enh_full.append(enh)
            enh_gev = istft(enh_gev, length=length)
            enh_gev = nc * enh_gev
            enh_gev_full.append(enh_gev)
        elif tlen > args.segments*int(tlen/args.segments):
            begin=args.segments*int(tlen/args.segments)
            #feats = feats_full[begin:,0,:]
            enh = feats_full[begin:,0]
            enh_full.append(enh)
            enh_gev = feats_full[begin:,0]
            enh_gev_full.append(enh_gev)


        enh_signal = np.hstack(enh_full)
        enh_gev_signal = np.hstack(enh_gev_full)
        for idx, name in enumerate(names):
            # Assuming mask, feats : [Batch, Time, Channel. Freq]
            #          enhanced    : [Batch, Time, Freq]
            #mask1 = ms1[idx][:ilens[idx]]
            #maskn = mn[idx][:ilens[idx]]
            #feat = feats[idx]['stft']
            #if 'pit' in train_args.bftype:
            #    mask2 = ms2[idx][:ilens[idx]]
            #    mask3 = ms3[idx][:ilens[idx]]
            #    mask4 = ms4[idx][:ilens[idx]]

            # Write enhanced wave files
            if enh_writer is not None:
                name_wav = name.split("_")[1] + 'MVDR'
                name_wav_gev = name.split("_")[1] + 'GEV'
                if args.enh_filetype in ('sound', 'sound.hdf5'):
                    enh_writer[name_wav] = (args.fs, enh_signal)
                    enh_writer[name_wav_gev] = (args.fs, enh_gev_signal)

def enhance(args):
    """Dumping enhanced speech and mask

    :param Namespace args: The program arguments
    """
    set_deterministic_pytorch(args)
    # read training config
    idim, train_args = get_model_conf(args.model, args.model_conf)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, train_args)
    torch_load(args.model, model)
    model.recog_args = args

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='separation', bftype=train_args.bftype, load_output=False, sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False})

    #load_inputs_and_targets = LoadInputsAndTargets(
    #    mode='asr', load_output=False, sort_in_input_length=False,
    #    preprocess_conf=None  # Apply pre_process in outer func
    #)
    if args.batchsize == 0:
        args.batchsize = 1

    # Creates writers for outputs from the network
    if args.enh_wspecifier is not None:
        enh_writer = FileWriterWrapper(args.enh_wspecifier,
                                       filetype=args.enh_filetype)
    else:
        enh_writer = None

    # Creates a Transformation instance
    preprocess_conf = (
        train_args.preprocess_conf if args.preprocess_conf is None
        else args.preprocess_conf)
    #if preprocess_conf is not None:
    #    logging.info('Use preprocessing'.format(preprocess_conf))
    #    transform = Transformation(preprocess_conf)
    #else:
    #    transform = None

    # Creates a IStft instance
    istft = None
    frame_shift = args.istft_n_shift  # Used for plot the spectrogram
    if args.apply_istft:
        if preprocess_conf is not None:
            # Read the conffile and find stft setting
            with open(preprocess_conf) as f:
                # Json format: e.g.
                #    {"process": [{"type": "stft",
                #                  "win_length": 400,
                #                  "n_fft": 512, "n_shift": 160,
                #                  "window": "han"},
                #                 {"type": "foo", ...}, ...]}
                conf = json.load(f)
                assert 'process' in conf, conf
                # Find stft setting
                for p in conf['process']:
                    if p['type'] == 'stft_pit':
                        istft = IStft(win_length=p['win_length'],
                                      n_shift=p['n_shift'],
                                      window=p.get('window', 'hann'))
                        logging.info('stft is found in {}. '
                                     'Setting istft config from it\n{}'
                                     .format(preprocess_conf, istft))
                        frame_shift = p['n_shift']
                        break
        if istft is None:
            # Set from command line arguments
            istft = IStft(win_length=args.istft_win_length,
                          n_shift=args.istft_n_shift,
                          window=args.istft_window)
            logging.info('Setting istft config from the command line args\n{}'
                         .format(istft))

    # sort data
    keys = list(js.keys())
    feat_lens = [js[key]['input'][0]['shape'][0] for key in keys]
    sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
    keys = [keys[i] for i in sorted_index]

    def grouper(n, iterable, fillvalue=None):
        kargs = [iter(iterable)] * n
        return zip_longest(*kargs, fillvalue=fillvalue)

    num_images = 0
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)

    for names in grouper(args.batchsize, keys, None):
        batch = [(name, js[name]) for name in names]

        # May be in time region: (Batch, [Time, Channel])
        feats = load_inputs_and_targets(batch)[0]
        #org_feats = load_inputs_and_targets(batch)[0]
        #if transform is not None:
        #    # May be in time-freq region: : (Batch, [Time, Channel, Freq])
        #    feats = transform(org_feats, train=False)
        #else:
        #    feats = org_feats

        with torch.no_grad():
            ms1, mn, ms2, ms3, ms4, enh, ilens = model.enhance(feats)

        for idx, name in enumerate(names):
            # Assuming mask, feats : [Batch, Time, Channel. Freq]
            #          enhanced    : [Batch, Time, Freq]
            mask1 = ms1[idx][:ilens[idx]]
            maskn = mn[idx][:ilens[idx]]
            feat = feats[idx]['stft']
            s1 = feats[idx]['s1']
            ns = feats[idx]['n']
            nc = feats[idx]['nc']
            enh1 = feat * mask1
            if 'pit' in train_args.bftype:
                mask2 = ms2[idx][:ilens[idx]]
                mask3 = ms3[idx][:ilens[idx]]
                mask4 = ms4[idx][:ilens[idx]]
                enh2 = feat * mask2
                enh3 = feat * mask3
                enh4 = feat * mask4
                s2 = feats[idx]['s2']
                s3 = feats[idx]['s3']
                s4 = feats[idx]['s4']

            # Plot spectrogram
            if args.image_dir is not None and num_images < args.num_images:
                import matplotlib.pyplot as plt
                plt.rcParams.update({'font.size': 20})
                plt.tight_layout()
                num_images += 1
                if 'pit' in train_args.bftype:
                    plt.clf()
                    plt.figure(figsize=(20, 15))
                    plt.subplot(4, 3, 1)
                    plt.title('Mixture')
                    plot_spectrogram(plt, np.abs(feat).T, fs=args.fs,
                                     mode='db', frame_shift=frame_shift,
                                     bottom=False, labelbottom=False)

                    plt.subplot(4, 3, 2)
                    plt.title('Source 1')
                    plot_spectrogram(plt, s1.T, fs=args.fs,
                                     mode='db', frame_shift=frame_shift,
                                     bottom=False, labelbottom=False)

                    plt.subplot(4, 3, 3)
                    plt.title('Mask 1')
                    plot_spectrogram(plt, mask1.T, fs=args.fs, mode='linear', frame_shift=frame_shift, bottom=False, labelbottom=False)

                    plt.subplot(4, 3, 4)
                    plt.title('Source 2')
                    plot_spectrogram(plt, s2.T, fs=args.fs,
                                     mode='db', frame_shift=frame_shift,
                                     bottom=False, labelbottom=False)

                    plt.subplot(4, 3, 5)
                    plt.title('Mask 2')
                    plot_spectrogram(plt, mask2.T, fs=args.fs, mode='linear', frame_shift=frame_shift)

                    plt.subplot(4, 3, 6)
                    plt.title('Source 3')
                    plot_spectrogram(plt, s3.T, fs=args.fs,
                                     mode='db', frame_shift=frame_shift,
                                     bottom=False, labelbottom=False)

                    plt.subplot(4, 3, 7)
                    plt.title('Mask 3')
                    plot_spectrogram(plt, mask3.T, fs=args.fs, mode='linear', frame_shift=frame_shift, bottom=False, labelbottom=False)

                    plt.subplot(4, 3, 8)
                    plt.title('Source 4')
                    plot_spectrogram(plt, s4.T, fs=args.fs,
                                     mode='db', frame_shift=frame_shift)

                    plt.subplot(4, 3, 9)
                    plt.title('Mask 4')
                    plot_spectrogram(plt, mask4.T, fs=args.fs, mode='linear', frame_shift=frame_shift, bottom=False, labelbottom=False)

                    plt.subplot(4, 3, 10)
                    plt.title('Noise')
                    plot_spectrogram(plt, ns.T, fs=args.fs,
                                     mode='db', frame_shift=frame_shift,
                                     bottom=False, labelbottom=False)

                    plt.subplot(4, 3, 11)
                    plt.title('Noise Mask')
                    plot_spectrogram(plt, maskn.T, fs=args.fs, mode='linear', frame_shift=frame_shift, bottom=False, labelbottom=False)

                    plt.subplot(4, 3, 12)
                    plt.title('Enhanced speech')
                    plot_spectrogram(plt, enh1.T, fs=args.fs, mode='db', frame_shift=frame_shift)

                    plt.savefig(os.path.join(args.image_dir, name + '.png'))
                else:
                    plt.clf()
                    plt.figure(figsize=(20, 10))
                    plt.subplot(5, 1, 1)
                    plt.title('Mixture')
                    plot_spectrogram(plt, np.abs(feat).T, fs=args.fs,
                                     mode='db', frame_shift=frame_shift,
                                     bottom=False, labelbottom=False)

                    plt.subplot(5, 1, 2)
                    plt.title('Source 1')
                    plot_spectrogram(plt, s1.T, fs=args.fs,
                                     mode='db', frame_shift=frame_shift,
                                     bottom=False, labelbottom=False)

                    plt.subplot(5, 1, 3)
                    plt.title('Mask 1')
                    plot_spectrogram(plt, mask1.T, fs=args.fs, mode='linear', frame_shift=frame_shift, bottom=False, labelbottom=False)

                    plt.subplot(5, 1, 4)
                    plt.title('Noise')
                    plot_spectrogram(plt, ns.T, fs=args.fs,
                                     mode='db', frame_shift=frame_shift,
                                     bottom=False, labelbottom=False)

                    plt.subplot(5, 1, 5)
                    plt.title('Noise Mask')
                    plot_spectrogram(plt, maskn.T, fs=args.fs, mode='linear', frame_shift=frame_shift, bottom=False, labelbottom=False)
                    plt.savefig(os.path.join(args.image_dir, name + '.png'))


            # Write enhanced wave files
            if enh_writer is not None:
                if istft is not None:
                    enh1 = istft(enh1)
                    enh = nc * enh1
                else:
                    enh = enh1

                #if args.keep_length:
                #    if len(org_feats[idx]) < len(enh):
                #        # Truncate the frames added by stft padding
                #        enh = enh[:len(org_feats[idx])]
                #    elif len(org_feats) > len(enh):
                #        padwidth = [(0, (len(org_feats[idx]) - len(enh)))] \
                #            + [(0, 0)] * (enh.ndim - 1)
                #        enh = np.pad(enh, padwidth, mode='constant')

                if args.enh_filetype in ('sound', 'sound.hdf5'):
                    enh_writer[name] = (args.fs, enh)
                else:
                    # Hint: To dump stft_signal, mask or etc,
                    # enh_filetype='hdf5' might be convenient.
                    enh_writer[name] = enh

            if num_images >= args.num_images and enh_writer is None:
                logging.info('Breaking the process.')
                break
