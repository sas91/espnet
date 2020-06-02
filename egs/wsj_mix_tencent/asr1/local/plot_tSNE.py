#!/usr/bin/env python

import argparse
from sklearn.decomposition import PCA
from matplotlib.font_manager import FontProperties
from sklearn.manifold import TSNE
import os
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.mlab as mlab
import numpy as np
import pickle
import pudb

def plot_manifold(X_r, conds, utt2cond, save=False, figname=None, xlim=None, ylim=None, s=8, aspect=0.6, fsize=12, ftype='png'):
    matplotlib.rcParams.update({'font.size': fsize, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans',
                                'text.usetex': False})

    cmap = cm.rainbow(np.linspace(0.0, 1.0, len(conds)))
    j = 0
    for cond in conds:
        indexes = []
        for i in range(len(utt2cond)):
            if utt2cond[i] == cond:
                indexes.append(i)
        colors = cmap[j]
        j = j+1
        plt.scatter(X_r[indexes, 0], X_r[indexes, 1], label=cond, marker='.', c=colors)

    fontP = FontProperties()

    axes = plt.gca()
    axes.set_aspect(aspect)
    axes.set_xlabel('dim 1')  # , fontname='Times New Roman', fontsize=fsize)
    axes.set_ylabel('dim 2')  # , fontname='Times New Roman', fontsize=fsize)
    # plt.axis('off')
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)

    plt.xticks([], [])
    plt.yticks([], [])
    # for tick in axes.get_xticklabels():
    #    tick.set_fontname('Times New Roman')
    # for tick in axes.get_yticklabels():
    #    tick.set_fontname('Times New Roman')

    fontP.set_size('small')
    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), ncol=1, shadow=False, scatterpoints=1, prop=fontP)

    if save:
        plt.savefig(figname + '.' + ftype, dpi=300, bbox_inches='tight')


def analyze_seqsum(exp, tset, root0='.', extension='.seqsum'):
    sqs = []
    spks = []
    utt2spk = []
    utt2spk_dic = {}
    utt2spk_file =  os.path.join(root0, 'data', tset, 'utt2spk')

    with open(utt2spk_file) as fin:
        for line in fin.readlines():
            utt, spk = line.split()
            spk = spk.split('_')[0]
            utt2spk_dic[utt] = spk

    for file in os.listdir(exp):
        if file.endswith(extension):
            spk = utt2spk_dic[file.split('.')[0]]
            if spk not in spks:
                spks.append(spk)
            utt2spk.append(spk)
            with open(exp + file, 'rb') as f:
                sq = pickle.load(f)
                sqs.append(sq[1])
    sqs_ar = np.array(sqs).transpose()

    return sqs_ar, spks, utt2spk


def pca_tsne(sqs_ar):
    X=sqs_ar.transpose()
    pca = PCA(n_components=2)
    X_pca = pca.fit(X).transform(X)
    X_tsne = TSNE(n_components=2).fit_transform(X)

    return X_pca, X_tsne


parser = argparse.ArgumentParser(description='Plot tSNE for Sequence Summary Network Vector')
#parser.add_argument('--exp', '-x', default='exp/wsj_si_tr_s_multich_pytorch_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.2_adadelta_sampprob0.0_bs20_mli600_mlo150_ng1_btmvdr_ss_rc-1_lsmunigram0.05_apt/enhance_wsj_et05_LN_multich/images/', type=str, help='Exp Directory')
#parser.add_argument('--exp', '-x', default='exp/wsj_si_tr_s_multich_pytorch_vggblstmp_e6_subsample1_2_2_1_1_unit320_proj320_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.2_adadelta_sampprob0.0_bs20_mli600_mlo150_ng1_btgdr_angle_ss_rc-1_lsmunigram0.05_apt/enhance_wsj_et05_LN_multich/images/', type=str, help='Exp Directory')
parser.add_argument('--exp', '-x', default='/data1/v_aswin/espnet/egs/wsj_mix/asr1_ts_multich/exp_enhance/enhance_wsj_si_tr_s_multich_pytorch_mse_bs100_ng1_btmask_angle_ss/enhance_wsj_et05_LN_multich_bf/images/', type=str, help='Exp Directory')
parser.add_argument('--ext', '-b', default='.seqsum', type=str,
                    help='Extension of SS vector files')
args = parser.parse_args()

tset = 'wsj_et05_LN_multich'
sqs_ar, spks, utt2spk = analyze_seqsum(args.exp, tset, extension=args.ext)
sqs_pca, sqs_tsne = pca_tsne(sqs_ar)

plt.figure(0)
if not os.path.exists(os.path.join(args.exp, 'visualize')):
    os.makedirs(os.path.join(args.exp, 'visualize'))
plot_manifold(sqs_tsne, spks, utt2spk, save=True,
              figname=os.path.join(args.exp, 'visualize', 'tsne_speaker'),
              s=10, aspect=1.2, fsize=14, ftype='png')
