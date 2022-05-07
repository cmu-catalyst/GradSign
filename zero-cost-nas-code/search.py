# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import pickle
import torch
import argparse
import time
import random
from statistics import mean, stdev

from foresight.models import *
from foresight.pruners import *
from foresight.dataset import *
from foresight.weight_initializers import init_net


def get_num_classes(dataset):
    return 100 if dataset == 'cifar100' else 10 if dataset == 'cifar10' else 120


def get_final_accuracy(args, api, uid, acc_type, trainval):
    if args.edataset == 'cifar10' and trainval:
        info = api.query_meta_info_by_index(uid, hp='200').get_metrics('cifar10-valid', 'x-valid')
    else:
        info = api.query_meta_info_by_index(uid, hp='200').get_metrics(args.edataset, acc_type)
    return info['accuracy']


def get_final_acc(dataset, api, uid, trainval):
    if dataset == 'cifar10':
        acc_type = 'ori-test'
        val_acc_type = 'x-valid'
    else:
        acc_type = 'x-test'
        val_acc_type = 'x-valid'

    if dataset == 'cifar10' and trainval:
        info = api.query_meta_info_by_index(uid, hp='200').get_metrics('cifar10-valid', 'x-valid')
    else:
        info = api.query_meta_info_by_index(uid, hp='200').get_metrics(dataset, acc_type)
    return info['accuracy']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Zero-cost Metrics for NAS-Bench-201')
    parser.add_argument('--api_loc', default='data/NAS-Bench-201-v1_0-e61699.pth',
                        type=str, help='path to API')
    parser.add_argument('--outdir', default='./search_201_results',
                        type=str, help='output directory')
    parser.add_argument('--method', default='synflow',
                        type=str, help='evaluating method name')
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--edataset', type=str, default='cifar10',
                        help='dataset to evaluate [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--sdataset', type=str, default='cifar10',
                        help='dataset to do the search [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1,
                        help='number of batches to use for random dataload or number of samples per class for grasp dataload')
    parser.add_argument('--seed', type=int, default=42, help='pytorch manual seed')
    parser.add_argument('--write_freq', type=int, default=10, help='frequency of write to file')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=0, help='end index')
    parser.add_argument('--runs', type=int, default=100, help='running number')
    parser.add_argument('--n_samples', type=int, default=100, help='sample size')
    parser.add_argument('--trainval', action='store_true')
    parser.add_argument('--noacc', default=False, action='store_true',
                        help='avoid loading NASBench2 api an instead load a pickle file with tuple (index, arch_str)')
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    return args


if __name__ == '__main__':
    args = parse_arguments()

    times = []
    chosen = []
    ret = {}
    test_acc = []
    val_acc = []
    test_acc_100 = []
    val_acc_100 = []
    test_acc_in = []
    val_acc_in = []
    topscores = []
    order_fn = np.nanargmax

    if args.edataset == 'cifar10':
        acc_type = 'ori-test'
        val_acc_type = 'x-valid'
    else:
        acc_type = 'x-test'
        val_acc_type = 'x-valid'

    if args.noacc:
        api = pickle.load(open(args.api_loc, 'rb'))
    else:
        from nas_201_api import NASBench201API as API
        api = API(args.api_loc)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader, val_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.sdataset,
                                                     args.num_data_workers)

    cached_res = []
    pre = 'cf' if 'cifar' in args.edataset else 'im'
    pfn = f'nb2_{pre}{get_num_classes(args.edataset)}_seed{args.seed}_dl{args.dataload}_dlinfo{args.dataload_info}_initw{args.init_w_type}_initb{args.init_b_type}.p'
    op = os.path.join(args.outdir, pfn)

    # if os.path.isfile(op):
    #     print("loading {}".format(op))
    #     with open(op, "rb") as fp:
    #         ret = pickle.load(fp)
    #         test_acc = ret["test_acc"]
    #         val_acc = ret["val_acc"]
    #         times = ret["times"]

    args.end = len(api) if args.end == 0 else args.end

    # loop over nasbench2 archs

    for run in range(args.runs):
        print("Evaluating Round: {}/{}".format(run+1, args.runs))

        t = 0
        indices = np.random.randint(0, args.end, args.n_samples)

        npstate = np.random.get_state()
        ranstate = random.getstate()
        torchstate = torch.random.get_rng_state()

        scores = []

        for i, arch in enumerate(indices):
            # print("Sample arch {}, {}/{}".format(arch, i+1, args.n_samples))
            arch_str = api[arch]
            net = nasbench2.get_model_from_arch_str(arch_str, get_num_classes(args.sdataset))
            net.to(args.device)

            init_net(net, args.init_w_type, args.init_b_type)
            start = time.time()
            measures = predictive.find_measures(net,
                                                train_loader,
                                                (args.dataload, args.dataload_info, get_num_classes(args.sdataset)),
                                                args.device,
                                                measure_names=[args.method])
            t += time.time() - start

            scores.append(measures[args.method])
            # exit()

        best_arch = indices[order_fn(scores)]
        # info = api.get_more_info(int(best_arch), 'cifar10-valid' if args.edataset == 'cifar10' else args.edataset, iepoch=None,
        #                          hp='200', is_random=False)
        info_100 = api.get_more_info(int(best_arch), 'cifar100', iepoch=None, hp='200', is_random=False)
        info_in = api.get_more_info(int(best_arch), 'ImageNet16-120', iepoch=None, hp='200', is_random=False)
        # acc_t = get_final_accuracy(args, api, int(best_arch), acc_type, False)
        acc_100 = get_final_acc('cifar100', api, int(best_arch), False)
        acc_in = get_final_acc('ImageNet16-120', api, int(best_arch), False)
        # uid = searchspace[best_arch]
        topscores.append(scores[order_fn(scores)])
        chosen.append(best_arch)
        # acc.append(searchspace.get_accuracy(uid, acc_type, args.trainval))
        # test_acc.append(acc_t)
        # val_acc.append(info['valid-accuracy'])
        test_acc_100.append(acc_100)
        val_acc_100.append(info_100['valid-accuracy'])
        test_acc_in.append(acc_in)
        val_acc_in.append(info_in['valid-accuracy'])

        times.append(t)
        if len(test_acc_100) > 2:
            print(f"mean cifar100 test acc: {mean(test_acc_100):.2f}+-{stdev(test_acc_100):.2f}, \
            mean val acc: {mean(val_acc_100):.2f}+-{stdev(val_acc_100):.2f}, \
            time:{mean(times):.2f}+-{stdev(times):.2f}")
            print(f"mean ImageNet16-120 test acc: {mean(test_acc_in):.2f}+-{stdev(test_acc_in):.2f}, \
                        mean val acc: {mean(val_acc_in):.2f}+-{stdev(val_acc_in):.2f}, \
                        time:{mean(times):.2f}+-{stdev(times):.2f}")

        if int(run+1) % args.write_freq == 0 or run+1 == args.runs:
            print(f'writing {len(test_acc)} results to {op}')
            fp = open(op, 'wb')
            ret["test_acc"] = test_acc
            ret["val_acc"] = val_acc
            ret["times"] = times
            pickle.dump(ret, fp)
            fp.close()

