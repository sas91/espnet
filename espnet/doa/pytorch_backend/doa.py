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

from chainer.datasets import TransformDataset
from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions
import numpy as np
from tensorboardX import SummaryWriter
import torch

from espnet.doa.asr_utils import adadelta_eps_decay
from espnet.doa.asr_utils import add_results_to_json
from espnet.doa.asr_utils import CompareValueTrigger
from espnet.doa.asr_utils import get_model_conf
from espnet.doa.asr_utils import make_batchset
from espnet.doa.asr_utils import plot_spectrogram
from espnet.doa.asr_utils import restore_snapshot
from espnet.doa.asr_utils import torch_load
from espnet.doa.asr_utils import torch_resume
from espnet.doa.asr_utils import torch_save
from espnet.doa.asr_utils import torch_snapshot
from espnet.nets.pytorch_backend.e2e_asr import pad_list
from espnet.transform.spectrogram import IStft
from espnet.transform.transformation import Transformation
from espnet.utils.cli_utils import FileWriterWrapper
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

    def __init__(self, subsampling_factor=1, loss_type='emd', resolution=1):
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.loss_type = loss_type
        self.resolution = resolution

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
        xs_p = [x['p'] for x in xs]
        ys_1 = [x['angle1'] for x in xs]
        ys_2 = [x['angle2'] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs_p])

        # perform padding and convert to tensor
        xs_pad = pad_list(
            [torch.from_numpy(x).float() for x in xs_p], 0).to(device)
        ys_pad_1 = torch.from_numpy(np.asarray(ys_1)/self.resolution).long().to(device)
        ys_pad_2 = torch.from_numpy(np.asarray(ys_2)/self.resolution).long().to(device)
        ys_pad = {'a1': ys_pad_1,
                  'a2': ys_pad_2}

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
    if 'ipd' in args.preprocess_conf:
        idim = int(valid_json[utts[0]]['input'][0]['shape'][-1]) * 2
    else:
        idim = int(valid_json[utts[0]]['input'][0]['shape'][-1])
    odim = 360/args.resolution
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # specify model architecture
    model_class = dynamic_import(args.model_module)
    model = model_class(idim, odim, args)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)),
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
    converter = CustomConverter(loss_type=args.loss_type,
                                resolution=args.resolution)

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
        mode='doa', preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode='doa', preprocess_conf=args.preprocess_conf,
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

    # Save best models
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


def classify(args):
    """Dumping enhanced speech and mask

    :param Namespace args: The program arguments
    """
    set_deterministic_pytorch(args)
    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, odim, train_args)
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
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='doa', sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False})

    if args.batchsize == 0:
        args.batchsize = 1

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
    # sort data
    keys = list(js.keys())
    feat_lens = [js[key]['input'][0]['shape'][0] for key in keys]
    sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
    keys = [keys[i] for i in sorted_index]

    def grouper(n, iterable, fillvalue=None):
        kargs = [iter(iterable)] * n
        return zip_longest(*kargs, fillvalue=fillvalue)

    total = 0.0
    correct = 0.0
    correct_m = 0.0
    correct_s = 0.0
    correct_w = 0.0
    correct_t = 0.0
    correct_f = 0.0
    prediction_error = 0.0
    prediction_error_m = 0.0
    prediction_error_s = 0.0
    prediction_error_w = 0.0
    prediction_error_t = 0.0
    prediction_error_f = 0.0
    for names in grouper(args.batchsize, keys, None):
        total += 1.0
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
            angle1, angle2, acc, acc_m, acc_s, acc_w, acc_t, acc_f, target1, target2, m1, m2, s1, s2, w1, w2, t1, t2, f1, f2 = model.classify(feats)
        music1_convert = m1 *args.resolution + (float(args.resolution)-1.0)/2.0
        music2_convert = m2 *args.resolution + (float(args.resolution)-1.0)/2.0
        srp1_convert = s1 *args.resolution + (float(args.resolution)-1.0)/2.0
        srp2_convert = s1 *args.resolution + (float(args.resolution)-1.0)/2.0
        waves1_convert = w1 *args.resolution + (float(args.resolution)-1.0)/2.0
        waves2_convert = w2 *args.resolution + (float(args.resolution)-1.0)/2.0
        tops1_convert = t1 *args.resolution + (float(args.resolution)-1.0)/2.0
        tops2_convert = t2 *args.resolution + (float(args.resolution)-1.0)/2.0
        frida1_convert = f1 *args.resolution + (float(args.resolution)-1.0)/2.0
        frida2_convert = f2 *args.resolution + (float(args.resolution)-1.0)/2.0
        angle1_convert = angle1 * args.resolution + (float(args.resolution)-1.0)/2.0
        angle2_convert = angle2 * args.resolution + (float(args.resolution)-1.0)/2.0
        target1_convert = target1 * args.resolution + (float(args.resolution)-1.0)/2.0
        target2_convert = target2 * args.resolution + (float(args.resolution)-1.0)/2.0
        prediction_error += np.minimum(
                              0.5 * (np.minimum(np.abs(angle1_convert-target1_convert),
                                        np.abs((angle1_convert+360)-target1_convert),
                                        np.abs(angle1_convert-(target1_convert+360))) + 
                              np.minimum(np.abs(angle2_convert-target2_convert),
                                        np.abs((angle2_convert+360)-target2_convert),
                                        np.abs(angle2_convert-(target2_convert+360)))),
                              0.5 * (np.minimum(np.abs(angle2_convert-target1_convert),
                                        np.abs((angle2_convert+360)-target1_convert),
                                        np.abs(angle2_convert-(target1_convert+360))) + 
                              np.minimum(np.abs(angle1_convert-target2_convert),
                                        np.abs((angle1_convert+360)-target2_convert),
                                        np.abs(angle1_convert-(target2_convert+360)))))
        prediction_error_m += np.minimum(
                              0.5 * (np.minimum(np.abs(music1_convert-target1_convert),
                                        np.abs((music1_convert+360)-target1_convert),
                                        np.abs(music1_convert-(target1_convert+360))) + 
                              np.minimum(np.abs(music2_convert-target2_convert),
                                        np.abs((music2_convert+360)-target2_convert),
                                        np.abs(music2_convert-(target2_convert+360)))),
                              0.5 * (np.minimum(np.abs(music2_convert-target1_convert),
                                        np.abs((music2_convert+360)-target1_convert),
                                        np.abs(music2_convert-(target1_convert+360))) + 
                              np.minimum(np.abs(music1_convert-target2_convert),
                                        np.abs((music1_convert+360)-target2_convert),
                                        np.abs(music1_convert-(target2_convert+360)))))
        prediction_error_s += np.minimum(
                              0.5 * (np.minimum(np.abs(srp1_convert-target1_convert),
                                        np.abs((srp1_convert+360)-target1_convert),
                                        np.abs(srp1_convert-(target1_convert+360))) + 
                              np.minimum(np.abs(srp2_convert-target2_convert),
                                        np.abs((srp2_convert+360)-target2_convert),
                                        np.abs(srp2_convert-(target2_convert+360)))),
                              0.5 * (np.minimum(np.abs(srp2_convert-target1_convert),
                                        np.abs((srp2_convert+360)-target1_convert),
                                        np.abs(srp2_convert-(target1_convert+360))) + 
                              np.minimum(np.abs(srp1_convert-target2_convert),
                                        np.abs((srp1_convert+360)-target2_convert),
                                        np.abs(srp1_convert-(target2_convert+360)))))
        prediction_error_w += np.minimum(
                              0.5 * (np.minimum(np.abs(waves1_convert-target1_convert),
                                        np.abs((waves1_convert+360)-target1_convert),
                                        np.abs(waves1_convert-(target1_convert+360))) + 
                              np.minimum(np.abs(waves2_convert-target2_convert),
                                        np.abs((waves2_convert+360)-target2_convert),
                                        np.abs(waves2_convert-(target2_convert+360)))),
                              0.5 * (np.minimum(np.abs(waves2_convert-target1_convert),
                                        np.abs((waves2_convert+360)-target1_convert),
                                        np.abs(waves2_convert-(target1_convert+360))) + 
                              np.minimum(np.abs(waves1_convert-target2_convert),
                                        np.abs((waves1_convert+360)-target2_convert),
                                        np.abs(waves1_convert-(target2_convert+360)))))
        prediction_error_t += np.minimum(
                              0.5 * (np.minimum(np.abs(tops1_convert-target1_convert),
                                        np.abs((tops1_convert+360)-target1_convert),
                                        np.abs(tops1_convert-(target1_convert+360))) + 
                              np.minimum(np.abs(tops2_convert-target2_convert),
                                        np.abs((tops2_convert+360)-target2_convert),
                                        np.abs(tops2_convert-(target2_convert+360)))),
                              0.5 * (np.minimum(np.abs(tops2_convert-target1_convert),
                                        np.abs((tops2_convert+360)-target1_convert),
                                        np.abs(tops2_convert-(target1_convert+360))) + 
                              np.minimum(np.abs(tops1_convert-target2_convert),
                                        np.abs((tops1_convert+360)-target2_convert),
                                        np.abs(tops1_convert-(target2_convert+360)))))
        prediction_error_f += np.minimum(
                              0.5 * (np.minimum(np.abs(frida1_convert-target1_convert),
                                        np.abs((frida1_convert+360)-target1_convert),
                                        np.abs(frida1_convert-(target1_convert+360))) + 
                              np.minimum(np.abs(frida2_convert-target2_convert),
                                        np.abs((frida2_convert+360)-target2_convert),
                                        np.abs(frida2_convert-(target2_convert+360)))),
                              0.5 * (np.minimum(np.abs(frida2_convert-target1_convert),
                                        np.abs((frida2_convert+360)-target1_convert),
                                        np.abs(frida2_convert-(target1_convert+360))) + 
                              np.minimum(np.abs(frida1_convert-target2_convert),
                                        np.abs((frida1_convert+360)-target2_convert),
                                        np.abs(frida1_convert-(target2_convert+360)))))
        logging.info('Prediction 1: %s' % angle1)
        logging.info('Prediction 2: %s' % angle2)
        logging.info('Ground Truth 1: %s' % target1)
        logging.info('Ground Truth 2: %s' % target2)
        logging.info('Prediction 1 - Converted : %s' % angle1_convert)
        logging.info('Prediction 2 - Converted : %s' % angle2_convert)
        logging.info('Prediction 1 MUSIC - Converted : %s' % music1_convert)
        logging.info('Prediction 2 MUSIC - Converted : %s' % music2_convert)
        logging.info('Prediction 1 SRP - Converted : %s' % srp1_convert)
        logging.info('Prediction 2 SRP - Converted : %s' % srp2_convert)
        logging.info('Prediction 1 WAVES - Converted : %s' % waves1_convert)
        logging.info('Prediction 2 WAVES - Converted : %s' % waves2_convert)
        logging.info('Prediction 1 TOPS - Converted : %s' % tops1_convert)
        logging.info('Prediction 2 TOPS - Converted : %s' % tops2_convert)
        logging.info('Prediction 1 FRIDA - Converted : %s' % frida1_convert)
        logging.info('Prediction 2 FRIDA - Converted : %s' % frida2_convert)
        logging.info('Ground Truth 1 - Converted : %s' % target1_convert)
        logging.info('Ground Truth 2 - Converted : %s' % target2_convert)
        correct += acc
        correct_m += acc_m
        correct_s += acc_s
        correct_w += acc_w
        correct_t += acc_t
        correct_f += acc_f
        accuracy = correct/total
        accuracy_m = correct_m/total
        accuracy_s = correct_s/total
        accuracy_w = correct_w/total
        accuracy_t = correct_t/total
        accuracy_f = correct_f/total
        avg_error = prediction_error/total
        avg_error_m = prediction_error_m/total
        avg_error_s = prediction_error_s/total
        avg_error_w = prediction_error_w/total
        avg_error_t = prediction_error_t/total
        avg_error_f = prediction_error_f/total
        logging.info('Correct Predictions: %s' % correct)
        logging.info('Total Predictions: %s' % total)
        logging.info('Accuracy - CNN: %s' % accuracy)
        logging.info('Accuracy - MUSIC: %s' % accuracy_m)
        logging.info('Accuracy - SRP: %s' % accuracy_s)
        logging.info('Accuracy - WAVES: %s' % accuracy_w)
        logging.info('Accuracy - TOPS: %s' % accuracy_t)
        logging.info('Accuracy - FRIDA: %s' % accuracy_f)
        logging.info('Prediction Error - CNN: %s' % avg_error)
        logging.info('Prediction Error - MUSIC: %s' % avg_error_m)
        logging.info('Prediction Error - SRP: %s' % avg_error_s)
        logging.info('Prediction Error - WAVES: %s' % avg_error_w)
        logging.info('Prediction Error - TOPS: %s' % avg_error_t)
        logging.info('Prediction Error - FRIDA: %s' % avg_error_f)
