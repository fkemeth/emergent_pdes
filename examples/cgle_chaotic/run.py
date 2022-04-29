import os
import shutil
import configparser

import tqdm
import torch

import lpde
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tests

from dataset import Dataset
from utils import dmaps_transform

POINTS_W = 397.48499

plt.set_cmap('plasma')


def make_plot_paper(config):
    """Plot CGLE simulation results."""
    data = np.load('data/run_'+str(config["SYSTEM"]["n_train"])+'.npy')

    idxs = np.arange(int(config["SYSTEM"]["N"]))
    np.random.shuffle(idxs)

    xx = np.linspace(0, float(config["SYSTEM"]["L"]), int(config["SYSTEM"]["N"]), endpoint=False)
    tt = np.linspace(0, float(config["SYSTEM"]["tmax"])-float(config["SYSTEM"]["tmin"]),
                     int((float(config["SYSTEM"]["tmax"])-float(config["SYSTEM"]["tmin"]))/float(config["SYSTEM"]["dt"]))+1)

    fig = plt.figure(figsize=(POINTS_W/72, 0.9*POINTS_W/72))
    ax1 = fig.add_subplot(321)
    pl1 = ax1.pcolor(xx, tt[::2],
                     data[::2, :, 0], vmin=-1, vmax=1,
                     rasterized=True, cmap='plasma')
    ax1.set_xlabel('$x$', labelpad=-2)
    ax1.set_ylabel('$t$', labelpad=0)
    ax1.set_xlim((0, float(config["SYSTEM"]["L"])))
    ax1.set_ylim((0, float(config["SYSTEM"]["tmax"])-float(config["SYSTEM"]["tmin"])))
    cbar1 = plt.colorbar(pl1)
    cbar1.set_label('Re $W$', labelpad=-3)
    ax2 = fig.add_subplot(322)
    pl2 = ax2.pcolor(np.arange(int(config["SYSTEM"]["N"])), tt[::2],
                     data[::2, idxs, 0], vmin=-1, vmax=1,
                     rasterized=True, cmap='plasma')
    ax2.set_xlabel('$i$', labelpad=-2)
    ax2.set_ylabel('$t$', labelpad=0)
    ax2.set_xlim((0, int(config["SYSTEM"]["N"])))
    ax2.set_ylim((0, float(config["SYSTEM"]["tmax"])-float(config["SYSTEM"]["tmin"])))
    cbar2 = plt.colorbar(pl2)
    cbar2.set_label('Re $W$', labelpad=-3)

    ax3 = fig.add_subplot(323)
    evecs = np.load('data/evecs.npy')
    # v_scaled = np.load(config["GENERAL"]["save_dir"]+'/v_scaled.npy')
    pl3 = ax3.scatter(evecs[:, 1], evecs[:, 2], s=2, c=xx[idxs],
                      cmap='plasma')

    ax3.set_xlabel(r'$\phi_1$', labelpad=-2)
    # ax3.set_xlim((0, con["N"]))
    ax3.set_ylabel(r'$\phi_2$', labelpad=-3)
    cbar3 = plt.colorbar(pl3)
    cbar3.set_label('$x$', labelpad=0)
    ax4 = fig.add_subplot(324)
    v_scaled = np.angle(evecs[:, 1]+1.0j*evecs[:, 2])
    pl4 = ax4.scatter(evecs[:, 1], evecs[:, 2], s=2, c=v_scaled,
                      cmap='plasma')
    ax4.set_xlabel(r'$\phi_1$', labelpad=-2)
    ax4.set_ylabel(r'$\phi_2$', labelpad=-3)
    cbar4 = plt.colorbar(pl4)
    cbar4.set_label(r'$\tilde{x}$', labelpad=0)

    ax5 = fig.add_subplot(325)
    pl5 = ax5.pcolor(np.roll(v_scaled, -np.argmin(v_scaled)), tt[::2],
                     np.roll(data[1::2, :, 0], -np.argmin(v_scaled), axis=1), vmin=-1, vmax=1,
                     rasterized=True, cmap='plasma')
    ax5.set_ylim((0, float(config["SYSTEM"]["tmax"])-float(config["SYSTEM"]["tmin"])))
    ax5.set_xlabel(r'$\tilde{x}$', labelpad=0)
    ax5.set_xlim((-np.pi, np.pi))
    ax5.set_ylabel(r'$t$', labelpad=0)
    cbar5 = plt.colorbar(pl5)
    cbar5.set_label('Re $W$', labelpad=-3)

    config.set("SYSTEM", "load_data", "True")
    config.set("SYSTEM", "L_orig", config["SYSTEM"]["L"])
    config.set("SYSTEM", "L", str(2*np.pi))
    dataset_train = Dataset(config["SYSTEM"], int(config["SYSTEM"]["n_train"]),
                            downsample=config["TRAINING"].getboolean('downsample'))
    dataset_test = Dataset(config["SYSTEM"], int(config["SYSTEM"]["n_test"]),
                           start_idx=int(config["SYSTEM"]["n_train"]))

    # Create Dataloader
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=int(config["TRAINING"]['batch_size']), shuffle=True,
        num_workers=int(config["TRAINING"]['num_workers']), pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=int(config["TRAINING"]['batch_size']), shuffle=False,
        num_workers=int(config["TRAINING"]['num_workers']), pin_memory=True)

    # Create the network architecture
    network = lpde.network.Network(config["MODEL"], n_vars=dataset_train.x_data.shape[1])

    model = lpde.model.Model(dataloader_train, dataloader_test, network,
                             config["TRAINING"], path=config["GENERAL"]["save_dir"])
    model.load_network(config["SYSTEM"]["boundary_conditions"]+'test.model')

    num_pars = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    print(num_pars)

    initial_condition, delta_x, _ = dataset_test[0]

    _, prediction = model.integrate(initial_condition.detach().numpy(), [
        delta_x.detach().numpy()], tt)

    ax6 = fig.add_subplot(326)
    pl6 = ax6.pcolor(np.roll(v_scaled, -np.argmin(v_scaled)), tt[::2],
                     np.roll(prediction[::2, 0], -np.argmin(v_scaled), axis=1),
                     vmin=-1, vmax=1, rasterized=True)
    ax6.set_xlabel(r'$\tilde{x}$', labelpad=0)
    ax6.set_ylabel(r'$t$', labelpad=0)
    ax6.set_xlim((-np.pi, np.pi))
    ax6.set_ylim((0, float(config["SYSTEM"]["tmax"])-float(config["SYSTEM"]["tmin"])))
    cbar6 = plt.colorbar(pl6)
    cbar6.set_label('Re $W$', labelpad=-3)
    ax1.text(-0.25, 1., r'$\mathbf{a}$', transform=ax1.transAxes, weight='bold', fontsize=12)
    ax2.text(-0.25,  1., r'$\mathbf{b}$', transform=ax2.transAxes, weight='bold', fontsize=12)
    ax3.text(-0.25, 1., r'$\mathbf{c}$', transform=ax3.transAxes, weight='bold', fontsize=12)
    ax4.text(-0.25,  1., r'$\mathbf{d}$', transform=ax4.transAxes, weight='bold', fontsize=12)
    ax5.text(-0.25, 1., r'$\mathbf{e}$', transform=ax5.transAxes, weight='bold', fontsize=12)
    ax6.text(-0.25,  1., r'$\mathbf{f}$', transform=ax6.transAxes, weight='bold', fontsize=12)
    plt.subplots_adjust(top=0.96, wspace=0.35, right=0.95, bottom=0.09, hspace=0.31, left=0.1)
    plt.show()

    np.save('Source_Data_Figure_2f.npy', prediction)
    np.save('Source_Data_Figure_2.npy', data)


def main(config):
    """Integrate system and train model."""

    if config["TRAINING"].getboolean("dtype64"):
        torch.set_default_dtype(torch.float64)

    verbose = config["GENERAL"]["verbose"]

    # Create Dataset
    dataset_train = Dataset(config["SYSTEM"], int(config["SYSTEM"]["n_train"]),
                            downsample=config["TRAINING"].getboolean('downsample'))
    dataset_test = Dataset(config["SYSTEM"], int(config["SYSTEM"]["n_test"]),
                           start_idx=int(config["SYSTEM"]["n_train"]))

    if verbose:
        tests.visualize_dynamics(dataset_train, path=config["GENERAL"]["fig_path"])
        # tests.visualize_dudt(dataset_train, path=config["GENERAL"]["fig_path"])

    # Create emergent space data
    if config["GENERAL"].getboolean("use_dmaps") and not config["SYSTEM"].getboolean("load_data"):
        dmaps_transform(int(config["SYSTEM"]["n_train"]) +
                        int(config["SYSTEM"]["n_test"]), dataset_train)
        config.set("SYSTEM", "load_data", "True")
        config.set("SYSTEM", "L_orig", config["SYSTEM"]["L"])
        config.set("SYSTEM", "L", str(2*np.pi))
        # config["SYSTEM"]["load_data"] = True
        # config["SYSTEM"]["L_orig"] = float(config["SYSTEM"]["L"])
        # config["SYSTEM"]["L"] = 2*np.pi
        dataset_train = Dataset(config["SYSTEM"], int(config["SYSTEM"]["n_train"]),
                                downsample=config["TRAINING"].getboolean('downsample'))
        dataset_test = Dataset(config["SYSTEM"], int(config["SYSTEM"]["n_test"]),
                               start_idx=int(config["SYSTEM"]["n_train"]))

    # Create Dataloader
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=int(config["TRAINING"]['batch_size']), shuffle=True,
        num_workers=int(config["TRAINING"]['num_workers']), pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=int(config["TRAINING"]['batch_size']), shuffle=False,
        num_workers=int(config["TRAINING"]['num_workers']), pin_memory=True)

    # Create the network architecture
    network = lpde.network.Network(config["MODEL"], n_vars=dataset_train.x_data.shape[1])

    if verbose:
        tests.visualize_derivatives(network, dataset_train, path=config["GENERAL"]["fig_path"])

    # Create a model wrapper around the network architecture
    # Contains functions for training
    model = lpde.model.Model(dataloader_train, dataloader_test, network,
                             config["TRAINING"], path=config["GENERAL"]["save_dir"])

    logger = SummaryWriter(config["GENERAL"]["save_dir"]+'/log/')

    progress_bar = tqdm.tqdm(range(0, int(config["TRAINING"]['epochs'])),
                             total=int(config["TRAINING"]['epochs']),
                             leave=True, desc=lpde.utils.progress(0, 0))

    # Load an already trained model if desired
    if config["TRAINING"].getboolean('proceed_training'):
        model.load_network(config["SYSTEM"]["boundary_conditions"]+'test.model')

    # Train the model
    train_loss_list = []
    val_loss_list = []
    for epoch in progress_bar:
        train_loss = model.train()
        val_loss = model.validate()
        progress_bar.set_description(lpde.utils.progress(train_loss, val_loss))

        logger.add_scalar('Loss/train', train_loss, epoch)
        logger.add_scalar('Loss/val', val_loss, epoch)
        logger.add_scalar('learning rate', model.optimizer.param_groups[-1]["lr"], epoch)

        model.save_network(config["SYSTEM"]["boundary_conditions"]+'test.model')

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    # Plot the loss curves
    if int(config["TRAINING"]['epochs']) > 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(train_loss_list, label='training loss')
        ax.plot(val_loss_list, label='validation loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel('')
        ax.set_yscale('log')
        plt.savefig(config["GENERAL"]["fig_path"]+'loss_curves.pdf')
        plt.show()

    # Plot the learned dudt
    tests.visualize_learned_dudt(
        dataset_test, model, path=config["GENERAL"]["fig_path"], epoch=None)

    # Visualize the predictions of the model
    # tests.visualize_predictions(dataset_test, model, path=config["GENERAL"]["fig_path"])


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config/config.cfg')
    main(config)

    make_plot_paper(config)
