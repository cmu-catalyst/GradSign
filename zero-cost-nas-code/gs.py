import os, pickle, sys
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import random


class GsApi:
    def __init__(self, api_loc):
        self.res = []
        print("Loading GradSign API from: {}".format(api_loc))
        f = open(api_loc, 'rb')
        while (1):
            try:
                self.res.append(pickle.load(f))
            except EOFError:
                break
        f.close()
        print("Done!")

    def __getitem__(self, item: int):
        return self.res[item]

    def __len__(self):
        return len(self.res)

    def get_score_by_index(self, index):
        assert index == self.res[index]["i"]
        return self.res[index]["score"]

    def get_time_by_index(self, index):
        assert index == self.res[index]["i"]
        return self.res[index]["time"]

    def get_acc_by_index(self, index):
        return self.res[index]["testacc"]


if __name__ == "__main__":
    fontsize = 23
    # CIFAR-10
    gs_api = GsApi("./201_results_batch_128/sum/nb2_cf10_seed42_base.p")
    print("{} has finished for cifar 10".format(len(gs_api)))
    acc = []
    score = []
    for i in range(len(gs_api)):
        acc.append(gs_api.get_acc_by_index(i))
        score.append(gs_api.get_score_by_index(i))
    print(stats.spearmanr(acc, score, nan_policy='omit').correlation)
    print("Generating correlation plot")
    sample_num = 1000
    index = random.sample(range(len(gs_api)), k=sample_num)
    # color = {"CIFAR10": "r", "CIFAR100": "g", "ImageNet16-120": "b"}
    plt.scatter(np.array(score)[index], np.array(acc)[index], color='r')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("GradSign metric score", fontsize=fontsize)
    plt.ylabel("Model accuracy", fontsize=fontsize)
    plt.title("{}".format("CIFAR-10"), fontsize=fontsize)
    plt.savefig("./CIFAR-10.pdf", bbox_inches="tight", dpi=500)
    plt.clf()

    # CIFAR-100
    gs_api = GsApi("./201_results_batch_128/sum/nb2_cf100_seed42_base.p")
    print("{} has finished for cifar 100".format(len(gs_api)))
    acc = []
    score = []
    for i in range(len(gs_api)):
        acc.append(gs_api.get_acc_by_index(i))
        score.append(gs_api.get_score_by_index(i))
    print(stats.spearmanr(acc, score, nan_policy='omit').correlation)
    print("Generating correlation plot")
    sample_num = 1000
    index = random.sample(range(len(gs_api)), k=sample_num)
    # color = {"CIFAR10": "r", "CIFAR100": "g", "ImageNet16-120": "b"}
    plt.scatter(np.array(score)[index], np.array(acc)[index], color='g')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("GradSign metric score", fontsize=fontsize)
    plt.ylabel("Model accuracy", fontsize=fontsize)
    plt.title("{}".format("CIFAR-100"), fontsize=fontsize)
    plt.savefig("./CIFAR-100.pdf", bbox_inches="tight", dpi=500)
    plt.clf()

    # ImageNet16-120
    gs_api = GsApi("./201_results_batch_128/sum/nb2_im120_seed42_base.p")
    print("{} has finished for imagenet16-120".format(len(gs_api)))
    acc = []
    score = []
    for i in range(len(gs_api)):
        acc.append(gs_api.get_acc_by_index(i))
        score.append(gs_api.get_score_by_index(i))
    print(stats.spearmanr(acc, score, nan_policy='omit').correlation)
    print("Generating correlation plot")
    sample_num = 1000
    index = random.sample(range(len(gs_api)), k=sample_num)
    # color = {"CIFAR10": "r", "CIFAR100": "g", "ImageNet16-120": "b"}
    plt.scatter(np.array(score)[index], np.array(acc)[index], color='b')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("GradSign metric score", fontsize=fontsize)
    plt.ylabel("Model accuracy", fontsize=fontsize)
    plt.title("{}".format("ImageNet16-120"), fontsize=fontsize)
    plt.savefig("./ImageNet16-120.pdf", bbox_inches="tight", dpi=500)
