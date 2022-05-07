import os, pickle, sys
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import glob
from prettytable import PrettyTable
from tqdm import tqdm

t = None
all_ds = {}
all_acc = {}
allc = {}
all_metrics = {}
all_runs = {}
metric_names = ['grad_conflict']
for fname, rname in [('./201_results/nb2_cf10_seed10_dlrandom_dlinfo1_initwnone_initbnone.p','CIFAR10'),
                     ('./201_results/nb2_cf100_seed10_dlrandom_dlinfo1_initwnone_initbnone.p', 'CIFAR100'),
                     ('./201_results/nb2_im120_seed10_dlrandom_dlinfo1_initwnone_initbnone.p', 'ImageNet16-120')
                     ]:
    runs = []
    f = open(fname, 'rb')
    while (1):
        try:
            runs.append(pickle.load(f))
        except EOFError:
            break
    f.close()
    print(fname, len(runs))

    all_runs[fname] = runs
    all_ds[fname] = {}
    metrics = {}
    for k in metric_names:
        metrics[k] = []
    acc = []

    if t is None:
        hl = ['Dataset']
        hl.extend(metric_names)
        t = PrettyTable(hl)

    for r in runs:
        for k, v in r['logmeasures'].items():
            if k in metrics:
                metrics[k].append(v)
        acc.append(r['testacc'])

    all_ds[fname]['metrics'] = metrics
    all_ds[fname]['acc'] = acc

    res = []
    crs = {}
    for k in hl:
        if k == 'Dataset':
            continue
        v = metrics[k]
        cr = abs(stats.spearmanr(acc, v, nan_policy='omit').correlation)
        # print(f'{k} = {cr}')
        res.append(round(cr, 3))
        crs[k] = cr

    ds = rname
    all_acc[ds] = acc
    allc[ds] = crs
    t.add_row([ds] + res)
    all_metrics[ds] = metrics
print(t)