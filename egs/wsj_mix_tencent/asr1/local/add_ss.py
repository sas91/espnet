#!/usr/bin/env python
# encoding: utf-8

import argparse
import sys
import random

from collections import defaultdict
xvec = {}
spk2utt = defaultdict(list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add xvector embedding for each mixture from different utterance',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--feats-scp', type=str)
    parser.add_argument('--utt2spk', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--out2', type=str)
    parser.add_argument('--out3', type=str)

    args = parser.parse_args()
    out = open(args.out, 'w')
    out2 = open(args.out2, 'w')
    out3 = open(args.out3, 'w')

    nutt = 0
    with open(args.feats_scp) as fp:
        line = fp.readline()
        while line:
            uttid = line.split()[0]
            spkid = uttid[0:3]
            embed = line.split()[1]
            xvec[uttid]=embed
            spk2utt[spkid].append(uttid)
            line = fp.readline()
            nutt += 1

    with open(args.utt2spk) as fp:
        line = fp.readline()
        while line:
            uttid = line.split()[0]
            uttid_target = line.split()[0].split('_')[2]
            spkid_target = line.split()[0].split('_')[0]
            embed_id = random.choice(spk2utt[spkid_target])
            while embed_id == uttid_target:
                embed_id = random.choice(spk2utt[spkid_target])
            embed_id_2 = random.choice(spk2utt[spkid_target])
            while (embed_id_2 == uttid_target) or (embed_id_2 == embed_id):
                embed_id_2 = random.choice(spk2utt[spkid_target])
            embed_id_3 = random.choice(spk2utt[spkid_target])
            while (embed_id_3 == uttid_target) or (embed_id_3 == embed_id) or (embed_id_3 == embed_id_2):
                embed_id_3 = random.choice(spk2utt[spkid_target])
            embed = xvec[embed_id]
            embed2 = xvec[embed_id_2]
            embed3 = xvec[embed_id_3]
            out.write('{} {}\n'.format(uttid, embed))
            out2.write('{} {}\n'.format(uttid, embed2))
            out3.write('{} {}\n'.format(uttid, embed3))
            line = fp.readline()
