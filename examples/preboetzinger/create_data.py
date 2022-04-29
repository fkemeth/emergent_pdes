import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pickle
import findiff

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
from scipy.integrate import ode
from scipy import interpolate

from utils import config, integrate_system, do_dmaps, dmaps_transform

from tests import test_perturbation


def main(config):
    integrate_system(config["DATA"], config["DATA"]["n_train"]+config["DATA"]["n_test"],
                     path=config["DATA"]['path'], verbose=config["GENERAL"]["verbose"])

    if config["GENERAL"]["verbose"]:
        test_perturbation(path=config["DATA"]["path"], idx=0)
        # tests.test_dt(mint.f, path=config["DATA"]["path"], idx=0)

    do_dmaps(config["DATA"], config["DATA"]["n_train"], path=config["DATA"]['path'], verbose=True)

    dmaps_transform(config["DATA"], config["DATA"]["n_train"]+config["DATA"]["n_test"],
                    path=config["DATA"]['path'], verbose=config["GENERAL"]["verbose"])


if __name__ == "__main__":
    main(config)
