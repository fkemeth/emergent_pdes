import os
import shutil
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

from torch.utils.tensorboard import SummaryWriter

from utils import config, Dataset, Network, Model, progress
from tests import test_perturbation


def main(config):
    dataset_train = Dataset(0, int(config["DATA"]["n_train"]), config["DATA"], config["MODEL"],
                            path=config["DATA"]["path"], use_svd=False)
    dataset_val = Dataset(int(config["DATA"]["n_train"]),
                          int(config["DATA"]["n_train"])+int(config["DATA"]["n_test"]),
                          config["DATA"], config["MODEL"],
                          path=config["DATA"]["path"], use_svd=False)

    # Create Dataloader
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=int(config["TRAINING"]['batch_size']), shuffle=True,
        num_workers=int(config["TRAINING"]['num_workers']), pin_memory=True)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(config["TRAINING"]['batch_size']), shuffle=False,
        num_workers=int(config["TRAINING"]['num_workers']), pin_memory=True)

    network = Network(config["MODEL"])

    model = Model(dataloader_train, dataloader_val, network, config["TRAINING"],
                  path=config["DATA"]["path"])

    if not os.path.exists(config["DATA"]["path"]+'log'):
        os.makedirs(config["DATA"]["path"]+'log')
    else:
        shutil.rmtree(config["DATA"]["path"]+'log')
        os.makedirs(config["DATA"]["path"]+'log')

    logger = SummaryWriter(config["DATA"]["path"]+'log/')

    progress_bar = tqdm.tqdm(range(0, int(config["TRAINING"]['epochs'])),
                             total=int(config["TRAINING"]['epochs']),
                             leave=True, desc=progress(0, 0))

    for epoch in progress_bar:
        train_loss = model.train()
        val_loss = model.validate()
        # model.store_model(val_accu, epoch)
        progress_bar .set_description(progress(train_loss, val_loss))

        logger.add_scalar('Loss/train', train_loss, epoch)
        logger.add_scalar('Loss/val', val_loss, epoch)
        logger.add_scalar('learning rate', model.optimizer.param_groups[-1]["lr"], epoch)

        model.save_network('test.model')


if __name__ == "__main__":
    main(config)
