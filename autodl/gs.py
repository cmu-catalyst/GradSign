import os, pickle, sys
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import glob
from prettytable import PrettyTable
from tqdm import tqdm


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
        self.score = []
        for k in self.res:
            self.score.append(k["logmeasures"]["grad_conflict"])
        self.sorted_score = np.sort(self.score).tolist()
        print("Done!")

    def __getitem__(self, item: int):
        return self.res[item]

    def __len__(self):
        return len(self.res)

    def get_score_by_index(self, index):
        assert index == self.res[index]["i"]
        return self.res[index]["logmeasures"]["grad_conflict"]

    def get_time_by_index(self, index):
        assert index == self.res[index]["i"]
        return self.res[index]["logmeasures"]["grad_conflict"]

    def in_top(self, score, rank):
        if score in self.sorted_score[-rank:]:
            return True
        else:
            return False

if __name__ == "__main__":
    gs_api = GsApi("./201_results_batch_128/mean/nb2_cf10_seed42_base.p")
    print("{} finished for cifar 10 mean".format(len(gs_api)))
    gs_api = GsApi("./201_results_batch_128/sum/nb2_cf10_seed42_base.p")
    print("{} finished for cifar 10 sum".format(len(gs_api)))
