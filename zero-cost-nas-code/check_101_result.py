import os, pickle, sys
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import glob
from tqdm import tqdm
from prettytable import PrettyTable

d = './101_results'
runs = []
processed = set()

for f in tqdm(os.listdir(d)):
    pf = open(os.path.join(d,f),'rb')
    while 1:
        try:
            p = pickle.load(pf)
            if p['hash'] in processed:
                continue
            processed.add(p['hash'])
            runs.append(p)
        except EOFError:
            break
    pf.close()
with open('./data/nasbench1_accuracy.p','rb') as f:
    all_accur = pickle.load(f)

t = None

print(d, len(runs))
metrics = {}
for k in runs[0]['logmeasures'].keys():
    metrics[k] = []
acc = []
hashes = []

if t is None:
    hl = ['Dataset']
    hl.extend(['grad_norm', 'snip', 'grasp', 'fisher', 'synflow', 'jacob_cov', 'grad_conflict', 'grad_angle'])
    t = PrettyTable(hl)

for r in runs:
    for k, v in r['logmeasures'].items():
        metrics[k].append(v)

    acc.append(all_accur[r['hash']][0])
    hashes.append(r['hash'])

res = []
for k in hl:
    if k == 'Dataset':
        continue
    v = metrics[k]
    cr = abs(stats.spearmanr(acc, v, nan_policy='omit').correlation)
    # print(f'{k} = {cr}')
    res.append(round(cr, 3))

ds = 'CIFAR10'
t.add_row([ds] + res)

print(t)