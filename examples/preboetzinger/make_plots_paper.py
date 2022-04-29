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
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import TruncatedSVD
from scipy.integrate import ode
from scipy import interpolate

import matplotlib.animation as animation

from torch.utils.tensorboard import SummaryWriter

from utils import config, Dataset, Network, Model, progress
from tests import test_perturbation


def main(config):
    dataset_train = Dataset(0, int(config["DATA"]["n_train"]), config["DATA"], config["MODEL"],
                            path=config["DATA"]["path"], use_svd=True)
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

    model.load_network('test.model')

    horizon = 5000
    prediction = model.integrate(dataset_val, dataset_train.svd, 0, horizon)

    X, Y = np.linspace(-1, 1, config["DATA"]["N"]), np.linspace(0, 1, config["DATA"]["N"])
    Xfull, Yfull = np.meshgrid(X, Y)

    t_arr = np.linspace(0, dataset_val.delta_t*horizon, horizon+1)

    def to_hV(data):
        # return data.real*15-37+1.0j*(data.imag*0.1+0.42)
        return np.stack((data[:, 0]*30-37, (data[:, 1]*0.2+0.42)), axis=1)

    pkl_file = open(config["DATA"]["path"]+str(0) + '_p_'+str(0)+'.pkl', 'rb')
    pkl_data = pickle.load(pkl_file)
    pkl_file.close()

    V = np.load(config["DATA"]["path"]+'dmaps_V.npy')
    df_data = {'snapshot_1_V': pkl_data["data"][500, :, 0],
               'snapshot_1_h': pkl_data["data"][500, :, 1],
               'snapshot_2_V': pkl_data["data"][1700, :, 0],
               'snapshot_2_h': pkl_data["data"][1700, :, 1],
               'snapshot_3_V': pkl_data["data"][2200, :, 0],
               'snapshot_3_h': pkl_data["data"][2200, :, 1],
               'snapshot_4_V': pkl_data["data"][2800, :, 0],
               'snapshot_4_h': pkl_data["data"][2800, :, 1],
               'snapshot_5_V': pkl_data["data"][3200, :, 0],
               'snapshot_5_h': pkl_data["data"][3200, :, 1],
               'phi_1': V[:, 1],
               'phi_2': V[:, 2],
               'I_app': pkl_data["xx"]
               }
    df = pd.DataFrame(df_data)
    df.to_excel(r'Source_Data_Figure_5.xlsx',
                sheet_name='Figure 5', index=False)

    POINTS_W = 397.48499
    fig = plt.figure(figsize=(POINTS_W/72, 5.5))
    ax1 = fig.add_subplot(321)
    ax1.plot(pkl_data["data"][::10, ::16, 0], pkl_data["data"][::10, ::16, 1], color='k', lw=0.3,
             zorder=1)
    idxs = [500, 1700, 2200, 2800, 3200]
    for i in idxs:
        # ax1.scatter(pkl_data["data"][i, ::16, 0], pkl_data["data"][i, ::16, 1],
        #             c=pkl_data["data"][i, ::16, 0], s=2, cmap='plasma', zorder=10)
        ax1.scatter(pkl_data["data"][i, ::16, 0], pkl_data["data"][i, ::16, 1],
                    s=2, cmap='plasma', zorder=10)
    ax1.set_xlabel(r'$V$', labelpad=0)
    ax1.set_ylabel(r'$h$')
    ax2 = fig.add_subplot(322)
    V = np.load(config["DATA"]["path"]+'dmaps_V.npy')
    v_scaled1 = V[:, 1]
    v_scaled2 = V[:, 2]
    xi, yi = np.meshgrid(np.linspace(-1, 1, config["DATA"]["N"]+1)[::2],
                         np.linspace(0, 1, config["DATA"]["N"]+1)[::2])
    ax2.plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
    ax2.plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
    scat2 = ax2.scatter(v_scaled1, v_scaled2, s=2, c=pkl_data["xx"], cmap='viridis')
    ax2.set_xlabel(r'$\phi_1$', labelpad=0)
    ax2.set_ylabel(r'$\phi_2$', labelpad=-3)
    cbar2 = plt.colorbar(scat2)
    cbar2.set_label(r'$I_{\mathrm{app}}^k$')

    ax3 = fig.add_subplot(323)
    pl3 = ax3.pcolor(X, Y, to_hV(dataset_val.x_data)[horizon+1, 0, :, :], vmin=-36, vmax=-28,
                     rasterized=True, cmap='plasma')
    ax3.set_xlabel(r'$\phi_1$', labelpad=0)
    ax3.set_ylabel(r'$\phi_2$')
    ax3.axvline(x=X[32], lw=1, color='k', linestyle='--')
    cbar3 = plt.colorbar(pl3, label='$V$')

    ax4 = fig.add_subplot(324)
    pl4 = ax4.pcolor(X, Y, to_hV(prediction)[-1, 0, :, :], vmin=-36, vmax=-28,
                     rasterized=True, cmap='plasma')
    ax4.set_xlabel(r'$\phi_1$', labelpad=0)
    ax4.set_ylabel(r'$\phi_2$', labelpad=0)
    cbar4 = plt.colorbar(pl4, label='$\hat{V}$')
    offset = dataset_train.off_set
    ax4.axvline(x=(X[offset]+X[offset+1])/2, ymin=(Y[offset]+Y[offset+1])/2,
                ymax=(Y[-1-offset]+Y[-2-offset])/2, color='white', lw=1)
    ax4.axvline(x=(X[-1-offset]+X[-2-offset])/2, ymin=(Y[offset]+Y[offset+1])/2,
                ymax=(Y[-1-offset]+Y[-2-offset])/2, color='white', lw=1)
    ax4.axhline(y=(Y[offset]+Y[offset+1])/2, xmin=((X[offset] + X[offset+1])/2+1)/2,
                xmax=((X[-1-offset]+X[-2-offset])/2+1)/2, color='white', lw=1)
    ax4.axhline(y=(Y[-1-offset]+Y[-2-offset])/2, xmin=((X[offset] + X[offset+1])/2+1)/2,
                xmax=((X[-1-offset]+X[-2-offset])/2+1)/2, color='white', lw=1)
    ax4.axvline(x=X[32], lw=1, color='k', linestyle='--')
    ax5 = fig.add_subplot(325)
    pl5 = ax5.pcolor(Y, t_arr[::4], to_hV(dataset_val.x_data)[
        :horizon+1:4, 0, :, 32], vmin=-50, vmax=-20, rasterized=True, cmap='plasma')
    ax5.set_xlabel(r'$\phi_2$', labelpad=0)
    ax5.set_ylabel('$t$', labelpad=0)
    cbar5 = plt.colorbar(pl5, label='$V$')
    ax6 = fig.add_subplot(326)
    pl6 = ax6.pcolor(Y, t_arr[::4], to_hV(prediction)[::4, 0, :, 32],
                     vmin=-50, vmax=-20, rasterized=True, cmap='plasma')
    ax6.set_xlabel(r'$\phi_2$', labelpad=0)
    ax6.set_ylabel('$t$', labelpad=0)
    cbar6 = plt.colorbar(pl6, label='$\hat{V}$')
    offset = dataset_train.off_set
    ax6.axvline(x=(Y[offset]+Y[offset+1])/2, ymin=0, ymax=1, color='white', lw=1)
    ax6.axvline(x=(Y[-1-offset]+Y[-2-offset])/2, ymin=0, ymax=1, color='white', lw=1)
    ax1.text(-0.23, 1., r'$\mathbf{a}$', transform=ax1.transAxes, weight='bold', fontsize=12)
    ax2.text(-0.3, 1., r'$\mathbf{b}$', transform=ax2.transAxes, weight='bold', fontsize=12)
    ax3.text(-0.3, 1., r'$\mathbf{c}$', transform=ax3.transAxes, weight='bold', fontsize=12)
    ax4.text(-0.3, 1., r'$\mathbf{d}$', transform=ax4.transAxes, weight='bold', fontsize=12)
    ax5.text(-0.3, 1., r'$\mathbf{e}$', transform=ax5.transAxes, weight='bold', fontsize=12)
    ax6.text(-0.3, 1., r'$\mathbf{f}$', transform=ax6.transAxes, weight='bold', fontsize=12)
    plt.subplots_adjust(top=0.97, wspace=0.4, right=0.92, bottom=0.08, left=0.1, hspace=0.3)
    # plt.savefig(config["DATA"]["path"] + 'lpde_hh_2d.pdf')
    # plt.savefig(config["DATA"]["path"] + 'lpde_hh_2d.png')
    plt.show()

    np.save('Source_Data_Figure_5c_and_e.npy', to_hV(dataset_val.x_data))
    np.save('Source_Data_Figure_5d_and_f.npy', to_hV(prediction))


if __name__ == "__main__":
    main(config)
