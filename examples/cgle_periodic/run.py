"""
Run CGLE example using specified config file.
"""
import int.cgle as cint
import tests
import lpde
import os
import pickle
import shutil
import configparser
import numpy as np
import matplotlib.pyplot as plt

import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

import utils_cgle

from scipy.spatial.distance import cdist


torch.set_default_dtype(torch.float32)


POINTS_W = 397.48499

plt.set_cmap('plasma')


def integrate_system(config, n, path, verbose=False, n_min=0):
    """Integrate complex Ginzburg-Landau equation."""
    pars = {}
    pars["c1"] = float(config["c1"])
    pars["c2"] = float(config["c2"])
    pars["c3"] = float(config["c3"])
    pars["mu"] = float(config["mu"])
    pars["L"] = float(config["L"])
    data_dict = cint.integrate(pars=pars,
                               dt=float(config["dt"]), N=int(config["N_int"]), T=int(config["T"]),
                               tmin=float(config["tmin"]), tmax=float(config["tmax"]),
                               append_init=True)

    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data_dict["xx"], data_dict["data"][-1].real, label='real')
        ax.plot(data_dict["xx"], data_dict["data"][-1].imag, label='imag')
        ax.set_xlabel(r'$\omega$')
        plt.title('snapshot')
        plt.legend()
        plt.show()

    for i in range(n_min, n):
        for p in [0, -1, 1]:
            data_perturbed = cint.integrate(pars=pars,
                                            dt=data_dict["dt"], N=data_dict["N"], T=data_dict["T"],
                                            tmin=0, tmax=data_dict["tmax"]-data_dict["tmin"],
                                            ic='manual',
                                            Ainit=data_dict["data"][int(i*int(config["T_off"]))] +
                                            p*float(config["eps"]) *
                                            data_dict["data"][int(i*int(config["T_off"]))],
                                            append_init=True)
            data_perturbed["data"] = data_perturbed["data"][:, ::int(
                int(config["N_int"])/int(config["N"]))]
            data_perturbed["xx"] = data_perturbed["xx"][::int(
                int(config["N_int"])/int(config["N"]))]
            data_perturbed["N"] = int(config["N"])
            output = open(path + 'run'+str(i)+'_p_'+str(p)+'.pkl', 'wb')
            pickle.dump(data_perturbed, output)
            output.close()


def make_plot_paper(config):
    """Plot CGLE simulation results."""
    pkl_file = open(config["GENERAL"]["save_dir"]+'/dat/run' +
                    config["TRAINING"]["n_train"]+'_p_'+str(0)+'.pkl', 'rb')
    data_dict = pickle.load(pkl_file)
    pkl_file.close()

    # t_off = 2000
    t_off = 0

    idxs = np.arange(data_dict["N"])
    np.random.shuffle(idxs)

    fig = plt.figure(figsize=(POINTS_W/72, 0.9*POINTS_W/72))
    ax1 = fig.add_subplot(321)
    pl1 = ax1.pcolor(data_dict["xx"], data_dict["tt"][::10]+t_off,
                     data_dict["data_org"][1::10].real, vmin=-1, vmax=1,
                     rasterized=True, cmap='plasma')
    ax1.set_xlabel('$x$', labelpad=-2)
    ax1.set_ylabel('$t$', labelpad=0)
    ax1.set_xlim((0, data_dict["L"]))
    ax1.set_ylim((data_dict["tmin"]+t_off, data_dict["tmax"]+t_off))
    cbar1 = plt.colorbar(pl1)
    cbar1.set_label('Re $W$', labelpad=-3)
    ax2 = fig.add_subplot(322)
    pl2 = ax2.pcolor(np.arange(data_dict["N"]), data_dict["tt"][::10]+t_off,
                     data_dict["data_org"][1::10, idxs].real, vmin=-1, vmax=1,
                     rasterized=True, cmap='plasma')
    ax2.set_xlabel('$i$', labelpad=-2)
    ax2.set_ylabel('$t$', labelpad=0)
    ax2.set_xlim((0, data_dict["N"]))
    ax2.set_ylim((data_dict["tmin"]+t_off, data_dict["tmax"]+t_off))
    cbar2 = plt.colorbar(pl2)
    cbar2.set_label('Re $W$', labelpad=-3)
    ax3 = fig.add_subplot(323)
    v_scaled = np.load(config["GENERAL"]["save_dir"]+'/v_scaled.npy')
    pl3 = ax3.scatter(np.arange(data_dict["N"]), v_scaled[idxs], s=2, c=data_dict["xx"][idxs],
                      cmap='plasma')

    ax3.set_xlabel('$i$', labelpad=-2)
    ax3.set_xlim((0, data_dict["N"]))
    ax3.set_ylabel(r'$\phi_1$', labelpad=-3)
    cbar3 = plt.colorbar(pl3)
    cbar3.set_label('$x$', labelpad=0)
    ax4 = fig.add_subplot(324)
    pl4 = ax4.pcolor(v_scaled, data_dict["tt"][::10]+t_off,
                     data_dict["data_org"][1::10].real, vmin=-1, vmax=1,
                     rasterized=True, cmap='plasma')
    ax4.set_ylim((data_dict["tmin"]+t_off, data_dict["tmax"]+t_off))
    ax4.set_xlabel(r'$\phi_1$', labelpad=0)
    ax4.set_xlim((-1, 1))
    ax4.set_ylabel(r'$t$', labelpad=0)
    cbar4 = plt.colorbar(pl4)
    cbar4.set_label('Re $W$', labelpad=-3)
    dataset_train = utils_cgle.Dataset(0, int(config["TRAINING"]["n_train"]), config["MODEL"],
                                       path=config["GENERAL"]["save_dir"])

    dataset_test = utils_cgle.Dataset(int(config["TRAINING"]["n_train"]),
                                      int(config["TRAINING"]["n_train"]) +
                                      int(config["TRAINING"]["n_test"]),
                                      config["MODEL"],
                                      path=config["GENERAL"]["save_dir"])

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=int(config["TRAINING"]['batch_size']), shuffle=True,
        num_workers=int(config["TRAINING"]['num_workers']), pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=int(config["TRAINING"]['batch_size']), shuffle=False,
        num_workers=int(config["TRAINING"]['num_workers']), pin_memory=True)

    network = lpde.network.Network(config["MODEL"], n_vars=2)

    model = lpde.model.Model(dataloader_train, dataloader_test, network, config["TRAINING"])

    model.load_network('test.model')
    num_pars = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    print(num_pars)

    pkl_file = open(config["GENERAL"]["save_dir"]+'/dat/run' +
                    config["TRAINING"]["n_train"]+'_p_'+str(0)+'.pkl', 'rb')
    data_unperturbed = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open(config["GENERAL"]["save_dir"]+'/dat/run' +
                    config["TRAINING"]["n_train"]+'_p_'+str(-1)+'.pkl', 'rb')
    data_perturbed_neg = pickle.load(pkl_file)
    pkl_file.close()

    prediction = model.integrate_svd(dataset_test, dataset_train.svd, 0, data_unperturbed["T"])

    print("Calculating closest distances....")
    dists_neg = cdist(np.append(data_perturbed_neg["data"].real, data_perturbed_neg["data"].imag,
                                axis=1), np.append(
        data_unperturbed["data"].real, data_unperturbed["data"].imag, axis=1))

    dists_learned = cdist(np.append(prediction[:, 0], prediction[:, 1], axis=1), np.append(
        data_unperturbed["data"].real, data_unperturbed["data"].imag, axis=1))

    phi_arr = np.linspace(-1, 1, data_unperturbed["N"])
    t_off = 0

    ax5 = fig.add_subplot(325)
    pl5 = ax5.pcolor(phi_arr, data_unperturbed["tt"][::10]+t_off,
                     prediction[1::10, 0], vmin=-1, vmax=1,
                     rasterized=True)
    ax5.axvline(x=(phi_arr[3]+phi_arr[4])/2, ymin=0, ymax=1, color='white', lw=1)
    ax5.axvline(x=(phi_arr[-4]+phi_arr[-5])/2, ymin=0, ymax=1, color='white', lw=1)
    ax5.set_xlabel(r'$\phi_1$', labelpad=0)
    ax5.set_ylabel(r'$t$', labelpad=0)
    ax5.set_xlim((-1, 1))
    ax5.set_ylim((data_unperturbed["tmin"]+t_off, data_unperturbed["tmax"]+t_off))
    cbar5 = plt.colorbar(pl5)
    cbar5.set_label('Re $W$', labelpad=-3)
    ax6 = fig.add_subplot(326)
    ax6.plot(data_unperturbed["tt"]+t_off, np.min(dists_neg, axis=1)[:-1], label='$d$ true')
    ax6.plot(data_unperturbed["tt"]+t_off, np.min(dists_learned, axis=1)
             [:-1], '--', label='$d$ learned')
    plt.legend()
    ax6.set_xlabel('$t$', labelpad=0)
    ax6.set_ylabel('$d$', labelpad=0)
    # plt.subplots_adjust(top=0.94, wspace=0.35, right=0.98, bottom=0.18, left=0.08)
    ax1.text(-0.25, 1., r'$\mathbf{a}$', transform=ax1.transAxes, weight='bold', fontsize=12)
    ax2.text(-0.25,  1., r'$\mathbf{b}$', transform=ax2.transAxes, weight='bold', fontsize=12)
    ax3.text(-0.25, 1., r'$\mathbf{c}$', transform=ax3.transAxes, weight='bold', fontsize=12)
    ax4.text(-0.25,  1., r'$\mathbf{d}$', transform=ax4.transAxes, weight='bold', fontsize=12)
    ax5.text(-0.25, 1., r'$\mathbf{e}$', transform=ax5.transAxes, weight='bold', fontsize=12)
    ax6.text(-0.25,  1., r'$\mathbf{f}$', transform=ax6.transAxes, weight='bold', fontsize=12)
    plt.subplots_adjust(top=0.96, wspace=0.35, right=0.95, bottom=0.09, hspace=0.31, left=0.08)
    plt.show()


def main(config):
    """Integrate system and train model."""

    verbose = config["GENERAL"].getboolean("verbose")

    # Create data folders
    if not os.path.exists(config["GENERAL"]["save_dir"]):
        os.makedirs(config["GENERAL"]["save_dir"])
    if not os.path.exists(config["GENERAL"]["save_dir"]+'/tests'):
        os.makedirs(config["GENERAL"]["save_dir"]+'/tests')

    # Create training and test data
    if not os.path.exists(config["GENERAL"]["save_dir"]+'/dat'):
        os.makedirs(config["GENERAL"]["save_dir"]+'/dat')
        if config["MODEL"].getboolean("use_param"):
            raise NotImplementedError
        else:
            integrate_system(config["SYSTEM"], int(config["TRAINING"]["n_train"]) +
                             int(config["TRAINING"]["n_test"]),
                             config["GENERAL"]["save_dir"]+'/dat/',
                             verbose=verbose)

    # Create Dataset
    dataset_train = utils_cgle.Dataset(0, int(config["TRAINING"]["n_train"]), config["MODEL"],
                                       path=config["GENERAL"]["save_dir"], verbose=verbose)
    dataset_test = utils_cgle.Dataset(int(config["TRAINING"]["n_train"]),
                                      int(config["TRAINING"]["n_train"]) +
                                      int(config["TRAINING"]["n_test"]),
                                      config["MODEL"],
                                      path=config["GENERAL"]["save_dir"], verbose=verbose)

    if config["GENERAL"].getboolean("use_dmaps"):
        utils_cgle.dmaps_transform(int(config["TRAINING"]["n_train"]) +
                                   int(config["TRAINING"]["n_test"]), dataset_train,
                                   path=config["GENERAL"]["save_dir"], verbose=verbose)
        dataset_train = utils_cgle.Dataset(0, int(config["TRAINING"]["n_train"]), config["MODEL"],
                                           path=config["GENERAL"]["save_dir"], verbose=verbose)
        dataset_test = utils_cgle.Dataset(int(config["TRAINING"]["n_train"]),
                                          int(config["TRAINING"]["n_train"]) +
                                          int(config["TRAINING"]["n_test"]),
                                          config["MODEL"],
                                          path=config["GENERAL"]["save_dir"], verbose=verbose)

    if verbose:
        tests.test_perturbation(path=config["GENERAL"]["save_dir"], idx=0)
        tests.test_dt(cint.f, path=config["GENERAL"]["save_dir"], idx=0)
        tests.test_dataset(dataset_train, path=config["GENERAL"]["save_dir"])
        if dataset.train.svd:
            tests.test_svd(dataset_train, dataset_test, path=config["GENERAL"]["save_dir"])

    # Create Dataloader
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=int(config["TRAINING"]['batch_size']), shuffle=True,
        num_workers=int(config["TRAINING"]['num_workers']), pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=int(config["TRAINING"]['batch_size']), shuffle=False,
        num_workers=int(config["TRAINING"]['num_workers']), pin_memory=True)

    network = lpde.network.Network(config["MODEL"], n_vars=2)

    delta_x = float(config["SYSTEM"]["L"])/int(config["SYSTEM"]["N"]) * \
        float(config["MODEL"]["rescale_dx"])

    if verbose:
        tests.test_fd_coeffs(network, path=config["GENERAL"]["save_dir"])
        tests.test_derivs(network, torch.tensor(dataset_train.x_data[:1],
                                                dtype=torch.get_default_dtype()),
                          torch.tensor([delta_x], dtype=torch.get_default_dtype()),
                          path=config["GENERAL"]["save_dir"])

    model = lpde.model.Model(dataloader_train, dataloader_test, network, config["TRAINING"])

    if not os.path.exists(config["GENERAL"]["save_dir"]+'/log'):
        os.makedirs(config["GENERAL"]["save_dir"]+'/log')
    else:
        shutil.rmtree(config["GENERAL"]["save_dir"]+'/log')
        os.makedirs(config["GENERAL"]["save_dir"]+'/log')

    logger = SummaryWriter(config["GENERAL"]["save_dir"]+'/log/')

    progress_bar = tqdm.tqdm(range(0, int(config["TRAINING"]['epochs'])),
                             total=int(config["TRAINING"]['epochs']),
                             leave=True, desc=lpde.utils.progress(0, 0))

    if config["GENERAL"].getboolean('proceed_training'):
        model.load_network('test.model')

    for epoch in progress_bar:
        train_loss = model.train()
        val_loss = model.validate()

        progress_bar.set_description(lpde.utils.progress(train_loss, val_loss))

        logger.add_scalar('Loss/train', train_loss, epoch)
        logger.add_scalar('Loss/val', val_loss, epoch)
        logger.add_scalar('learning rate', model.optimizer.param_groups[-1]["lr"], epoch)

        model.save_network('test.model')

    if verbose:
        model = lpde.model.Model(dataloader_train, dataloader_test, network, config["TRAINING"])
        model.load_network('test.model')
        tests.test_learned_dt(model, dataset_test, cint.f,
                              path=config["GENERAL"]["save_dir"], idx=0)
        tests.test_learned_dt(model, dataset_test, cint.f,
                              path=config["GENERAL"]["save_dir"], idx=2500)
        tests.test_learned_dt(model, dataset_test, cint.f,
                              path=config["GENERAL"]["save_dir"], idx=4500)
        _ = tests.test_integration(model, dataset_test, dataset_train.svd, 1000, 4000,
                                   path=config["GENERAL"]["save_dir"])

        tests.test_transient_dynamics(model, dataset_test, dataset_train.svd,
                                      idx=int(config["TRAINING"]["n_train"]), t_off=0,
                                      path=config["GENERAL"]["save_dir"])


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config/config.cfg')
    main(config)

    make_plot_paper(config)
