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
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import TruncatedSVD
from scipy.integrate import ode
from scipy import interpolate

import matplotlib.animation as animation

from torch.utils.tensorboard import SummaryWriter

from utils import config, Dataset, Network, Model, progress, from_hV, chung_lu, to_hV
from tests import test_perturbation

import int.twoConductanceEquations as twoConductanceEquations


def plot_data(config):
    if "T" not in config.keys():
        config = config["DATA"]
    T = np.int(config["T"])
    tmax = np.int(config["tmax"])
    tmin = np.int(config["tmin"])

    pars = {}
    pars["N"] = np.int(config["N_int"])
    pars["domain"] = 20+4*np.random.rand(pars["N"])
    pars["dx"] = pars["domain"][1]-pars["domain"][0]
    pars["A"] = chung_lu(pars["N"])

    X0 = twoConductanceEquations.DataGenerator(
        N=pars["N"], Iapp=pars["domain"], A=pars["A"],
        rtol=1e-6, atol=1e-9).simulator.history.y[:, -1]

    sim = twoConductanceEquations.TwoCondEqnSim(
        N=pars["N"], Iapp=pars["domain"], A=pars["A"])
    sim.integrate(X0=X0, t_span=[0, tmax],
                  t_eval=np.linspace(tmin, tmax, T+1), rtol=1e-8, atol=1e-10)

    data = sim.history.y

    fig = plt.figure(figsize=(9, 4.5))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'$h_i$')
    ax1.set_ylabel(r'$V_i$')
    ax1.set_xlim((-60, -10))
    ax1.set_ylim((0.28, 0.55))
    for i in range(64):
        ax1.plot(data[i, :], data[i+1024, :], lw=0.1, color='k')
    ax1.scatter(data[:1024, 0], data[1024:, 0], zorder=10)
    plt.show()

    print("Creating images. This may take a few seconds.")

    fig = plt.figure(figsize=(9, 4.5))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'$h_i$')
    ax1.set_ylabel(r'$V_i$')
    ax1.set_xlim((-60, -10))
    ax1.set_ylim((0.28, 0.55))
    # ax1.scatter(data[:1024, 0], data[1024:, 0], zorder=10, color='black', s=5)
    for i in range(0, 1024, 16):
        ax1.plot(data[i, :], data[i+1024, :], lw=0.1, color='k')
    scas = []
    for i in range(0, 10000, 10):
        sca1 = ax1.scatter(data[:1024, i], data[1024:, i], zorder=10, color='black', s=5)
        scas.append([sca1])
    ani = animation.ArtistAnimation(
        fig, scas, interval=200, blit=True, repeat_delay=0, repeat=True)
    fps = 100
    bitrate = 2000
    FFMpegWriter = animation.writers['ffmpeg']
    FFwriter = FFMpegWriter(fps=fps, bitrate=bitrate)
    ani.save('/home/felix/gtd/tex/dresden/interview/preboetzinger_lines.mp4', writer=FFwriter, dpi=200)
    plt.show()


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

    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(121)
    pl1 = ax1.pcolor(Y, X, to_hV(dataset_val.x_data)[horizon+1, 0, :, :], vmin=-36, vmax=-28,
                     rasterized=True)
    ax1.set_xlabel(r'$\phi_2$')
    ax1.set_ylabel(r'$\phi_1$')
    cbar1 = plt.colorbar(pl1, label='$h$')
    ax2 = fig.add_subplot(122)
    pl2 = ax2.pcolor(Y, X, to_hV(prediction)[-1, 0, :, :], vmin=-36, vmax=-28,
                     rasterized=True)
    ax2.set_xlabel(r'$\phi_2$')
    ax2.set_ylabel(r'$\phi_1$')
    cbar2 = plt.colorbar(pl2, label='$h$')
    offset = dataset_train.off_set
    ax2.axvline(x=(Y[offset]+Y[offset+1])/2, ymin=((X[offset]+X[offset+1])/2+1)/2,
                ymax=((X[-1-offset]+X[-2-offset])/2+1)/2, color='white', lw=1)
    ax2.axvline(x=(Y[-1-offset]+Y[-2-offset])/2, ymin=((X[offset]+X[offset+1])/2+1)/2,
                ymax=((X[-1-offset]+X[-2-offset])/2+1)/2, color='white', lw=1)
    ax2.axhline(y=(X[offset]+X[offset+1])/2, xmin=(Y[offset] + Y[offset+1])/2,
                xmax=(Y[-1-offset]+Y[-2-offset])/2, color='white', lw=1)
    ax2.axhline(y=(X[-1-offset]+X[-2-offset])/2, xmin=(Y[offset] + Y[offset+1])/2,
                xmax=(Y[-1-offset]+Y[-2-offset])/2, color='white', lw=1)
    ax2.axhline(y=X[32], lw=1, color='k', linestyle='--')
    plt.subplots_adjust(top=0.97, wspace=0.4, right=0.92, bottom=0.14, left=0.1)
    plt.savefig(config["DATA"]["path"] + 'predictions_snapshots.pdf')
    plt.savefig(config["DATA"]["path"] + 'predictions_snapshots.png')
    plt.show()

    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(121)
    pl1 = ax1.pcolor(Y, t_arr[::4], to_hV(dataset_val.x_data)[
                     :horizon+1:4, 0, :, 32], vmin=-60, vmax=-20, rasterized=True)
    ax1.set_xlabel(r'$\phi_2$')
    ax1.set_ylabel('$t$')
    cbar1 = plt.colorbar(pl1, label='$h$')
    ax2 = fig.add_subplot(122)
    pl2 = ax2.pcolor(Y, t_arr[::4], to_hV(prediction)[::4, 0, :, 32],
                     vmin=-60, vmax=-20, rasterized=True)
    ax2.set_xlabel(r'$\phi_2$')
    ax2.set_ylabel('$t$')
    cbar2 = plt.colorbar(pl2, label='$h$')
    offset = dataset_train.off_set
    ax2.axvline(x=(Y[offset]+Y[offset+1])/2, ymin=0, ymax=1, color='white', lw=1)
    ax2.axvline(x=(Y[-1-offset]+Y[-2-offset])/2, ymin=0, ymax=1, color='white', lw=1)
    plt.subplots_adjust(top=0.97, wspace=0.4, right=0.92, bottom=0.14, left=0.1)
    plt.savefig(config["DATA"]["path"] + 'predictions_spacetime.pdf')
    plt.savefig(config["DATA"]["path"] + 'predictions_spacetime.png')
    plt.show()

    V = np.load(config["DATA"]["path"]+'dmaps_V.npy')

    v_scaled1 = V[:, 1]
    v_scaled2 = V[:, 2]
    i = int(config["DATA"]["n_train"])
    p = config["DATA"]["p_list"][0]
    pkl_file = open(config["DATA"]["path"]+str(i) + '_p_'+str(p)+'.pkl', 'rb')
    pkl_data = pickle.load(pkl_file)
    pkl_file.close()

    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(121, projection='3d')
    sca1 = ax1.scatter(v_scaled1[v_scaled2 > 0],
                       v_scaled2[v_scaled2 > 0],
                       pkl_data["data"][horizon, v_scaled2 > 0, 0], zorder=10, color='black', s=5)

    surf1 = ax1.plot_surface(
        Xfull, Yfull, to_hV(np.transpose(pkl_data["data_fitted"], (0, 3, 1, 2)))[horizon, 0],
        alpha=0.5, color='blue')
    ax1.set_xlabel(r'$\phi_1$')
    ax1.set_ylabel(r'$\phi_2$')
    ax1.set_zlabel(r'$h$')
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(
        Xfull, Yfull, to_hV(prediction)[horizon, 0],
        alpha=0.5, color='blue')
    ax2.set_xlabel(r'$\phi_1$')
    ax2.set_ylabel(r'$\phi_2$')
    ax2.set_zlabel(r'$\hat{h}$')
    ax1.set_zlim((-60, -20))
    ax2.set_zlim((-60, -20))
    ax1.view_init(azim=139, elev=33)
    ax2.view_init(azim=139, elev=33)
    ax1.set_title('True and fitted')
    ax2.set_title('Prediction')
    plt.savefig(config["DATA"]["path"] + 'predictions_3dscatter.pdf')
    plt.savefig(config["DATA"]["path"] + 'predictions_3dscatter.png')
    plt.show()

    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlabel(r'$\phi_1$')
    ax1.set_ylabel(r'$\phi_2$')
    ax1.set_zlabel(r'$h$')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlabel(r'$\phi_1$')
    ax2.set_ylabel(r'$\phi_2$')
    ax2.set_zlabel(r'$\hat{h}$')
    ax1.set_zlim((-60, -20))
    ax2.set_zlim((-60, -20))
    ax1.view_init(azim=139, elev=33)
    ax2.view_init(azim=139, elev=33)
    ax1.set_title('True and fitted')
    ax2.set_title('Prediction')
    print("Creating images. This may take a few seconds.")
    scas = []
    for i in range(0, horizon, 50):
        sca1 = ax1.scatter(v_scaled1[v_scaled2 > 0],
                           v_scaled2[v_scaled2 > 0],
                           pkl_data["data"][i, v_scaled2 > 0, 0], zorder=10, color='black', s=5)
        surf1 = ax1.plot_surface(
            Xfull[::2, ::2], Yfull[::2, ::2],
            to_hV(np.transpose(pkl_data["data_fitted"], (0, 3, 1, 2)))[i, 0, ::2, ::2],
            alpha=0.5, color='blue')
        surf2 = ax2.plot_surface(
            Xfull[::2, ::2], Yfull[::2, ::2], to_hV(prediction)[i, 0, ::2, ::2],
            alpha=0.5, color='blue')
        scas.append([sca1, surf1, surf2])

    # ani = animation.ArtistAnimation(
    #     fig, scas, interval=400, blit=True, repeat_delay=0, repeat=True)
    # writergif = animation.PillowWriter(fps=20)
    # ani.save(config["DATA"]["path"] + 'predictions_vid_surf.gif', writer=writergif)
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=500, codec='h264')
    # ani.save(config["DATA"]["path"] + 'predictions_vid_surf.avi', writer=writer)
    ani = animation.ArtistAnimation(
        fig, scas, interval=400, blit=True, repeat_delay=0, repeat=True)
    fps = 20
    bitrate = 2000
    FFMpegWriter = animation.writers['ffmpeg']
    FFwriter = FFMpegWriter(fps=fps, bitrate=bitrate)
    ani.save('/home/felix/gtd/tex/dresden/interview/preboetzinger_predictions.mp4',
             writer=FFwriter, dpi=200)
    plt.show()


if __name__ == "__main__":
    main(config)
