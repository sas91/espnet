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
    parser.add_argument('--xvector-scp', type=str)
    parser.add_argument('--utt2spk', type=str)
    parser.add_argument('--out', type=str)

    args = parser.parse_args()
    out = open(args.out, 'w')

    nutt = 0
    with open(args.xvector_scp) as fp:
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
            embed = xvec[embed_id]
            out.write('{} {}\n'.format(uttid, embed))
            line = fp.readline()
