"""
Run Matthews example using specified config file.
"""
import os
import pickle
import shutil
import configparser
import numpy as np
import matplotlib.pyplot as plt

import tqdm
import torch
import lpde
from torch.utils.tensorboard import SummaryWriter

from scipy.spatial.distance import cdist

import utils
import tests

import int.matthews as mint

torch.set_default_dtype(torch.float32)

POINTS_W = 397.48499

plt.set_cmap('plasma')


def integrate_system(config, n, path, verbose=False):
    """Integrate Matthews system."""
    pars = {}
    pars["gamma"] = float(config["gamma"])
    pars["omega"] = np.linspace(-pars["gamma"], pars["gamma"], int(config["N_int"])) + \
        float(config["gamma_off"])
    pars["K"] = float(config["K"])
    data_dict = mint.integrate(pars=pars,
                               dt=float(config["dt"]), N=int(config["N_int"]), T=int(config["T"]),
                               tmin=float(config["tmin"]), tmax=float(config["tmax"]),
                               gamma_off=float(config["gamma_off"]),
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

    for i in range(n):
        perturbation = utils.perturb_limit_cycle(data_dict, i, config)

        if verbose:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(data_dict["xx"], perturbation.real, label='real')
            ax.plot(data_dict["xx"], perturbation.imag, label='imag')
            ax.set_xlabel(r'$\omega$')
            plt.title('perturbation')
            plt.legend()
            plt.show()

        for p in [0, -1, 1]:
            data_perturbed = mint.integrate(pars=pars,
                                            dt=data_dict["dt"], N=data_dict["N"], T=data_dict["T"],
                                            tmin=0, tmax=data_dict["tmax"]-data_dict["tmin"],
                                            ic='manual',
                                            Ainit=data_dict["data"][int(i*int(config["T_off"]))] +
                                            float(config["eps"])*p*perturbation,
                                            gamma_off=float(config["gamma_off"]),
                                            append_init=True)
            data_perturbed["data"] = data_perturbed["data"][:, ::int(
                int(config["N_int"])/int(config["N"]))]
            data_perturbed["omega"] = data_perturbed["omega"][::int(
                int(config["N_int"])/int(config["N"]))]
            data_perturbed["xx"] = data_perturbed["xx"][::int(
                int(config["N_int"])/int(config["N"]))]
            data_perturbed["N"] = int(config["N"])
            output = open(path + 'run'+str(i)+'_p_'+str(p)+'.pkl', 'wb')
            pickle.dump(data_perturbed, output)
            output.close()


def integrate_system_gamma(config, n, path, verbose=False):
    """Integrate Matthews system."""
    gamma_list = np.linspace(1.7, 1.8, n-5)
    gamma_list2 = np.append(1.7+0.1*np.random.random(4), 1.8)
    # np.random.shuffle(gamma_list)
    gamma_crit = 1.747

    for i, gamma in enumerate(np.append(gamma_list, gamma_list2)):
        print("gamma: "+str(gamma))
        pars = {}
        pars["gamma"] = gamma
        pars["omega"] = np.linspace(-pars["gamma"], pars["gamma"], int(config["N_int"])) + \
            float(config["gamma_off"])
        pars["K"] = float(config["K"])
        data_dict = mint.integrate(pars=pars,
                                   dt=float(config["dt"]), N=int(config["N_int"]), T=int(config["T"]),
                                   tmin=float(config["tmin"]), tmax=float(config["tmax"]),
                                   gamma_off=float(config["gamma_off"]),
                                   append_init=True)

        if gamma < gamma_crit:
            perturbation = utils.perturb_limit_cycle(data_dict, i, config)
        else:
            perturbation = utils.perturb_fixed_point(data_dict, i, config)

        if verbose:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(pars["omega"], data_dict["data"][-1].real, label='real')
            ax.plot(pars["omega"], data_dict["data"][-1].imag, label='imag')
            ax.set_xlabel(r'$\omega$')
            plt.title('snapshot')
            plt.legend()
            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(pars["omega"], perturbation.real, label='real')
            ax.plot(pars["omega"], perturbation.imag, label='imag')
            ax.set_xlabel(r'$\omega$')
            plt.title('perturbation')
            plt.legend()
            plt.show()

        for p in [0, -1, 1]:
            data_perturbed = mint.integrate(pars=pars,
                                            dt=data_dict["dt"], N=data_dict["N"], T=data_dict["T"],
                                            tmin=0, tmax=data_dict["tmax"]-data_dict["tmin"],
                                            ic='manual',
                                            Ainit=data_dict["data"][int(i*int(config["T_off"]))] +
                                            float(config["eps"])*p*perturbation,
                                            gamma_off=float(config["gamma_off"]),
                                            append_init=True)
            data_perturbed["data"] = data_perturbed["data"][:, ::int(
                int(config["N_int"])/int(config["N"]))]
            data_perturbed["omega"] = data_perturbed["omega"][::int(
                int(config["N_int"])/int(config["N"]))]
            data_perturbed["xx"] = data_perturbed["xx"][::int(
                int(config["N_int"])/int(config["N"]))]
            data_perturbed["N"] = int(config["N"])
            output = open(path + 'run'+str(i)+'_p_'+str(p)+'.pkl', 'wb')
            pickle.dump(data_perturbed, output)
            output.close()


def make_plot_paper(config):
    """Plot SLE learning results."""

    dataset_train = utils.Dataset(0, int(config["TRAINING"]["n_train"]),
                                  config["MODEL"],
                                  path=config["GENERAL"]["save_dir"])

    dataset = utils.Dataset(int(config["TRAINING"]["n_train"]),
                            int(config["TRAINING"]["n_train"]) +
                            int(config["TRAINING"]["n_test"]),
                            config["MODEL"],
                            path=config["GENERAL"]["save_dir"])

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=int(config["TRAINING"]['batch_size']), shuffle=True,
        num_workers=int(config["TRAINING"]['num_workers']), pin_memory=True)

    network = lpde.network.Network(config["MODEL"], n_vars=2)

    model = lpde.model.Model(dataloader, dataloader, network, config["TRAINING"],
                             path=config["GENERAL"]["save_dir"]+'/')
    model.load_network('test.model')

    pkl_file = open(config["GENERAL"]["save_dir"]+'/dat/run' +
                    str(int(config["TRAINING"]["n_train"])) + '_p_'+str(0)+'.pkl', 'rb')
    data_unperturbed = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open(config["GENERAL"]["save_dir"]+'/dat/run' +
                    str(int(config["TRAINING"]["n_train"])) + '_p_'+str(-1)+'.pkl', 'rb')
    data_perturbed_neg = pickle.load(pkl_file)
    pkl_file.close()

    t_off = 0

    prediction = model.integrate_svd(dataset, dataset_train.svd,
                                     t_off, data_unperturbed["T"]-1-t_off)

    print("Calculating closest distances....")
    dists_neg = cdist(np.append(data_perturbed_neg["data"].real, data_perturbed_neg["data"].imag,
                                axis=1), np.append(
        data_unperturbed["data"].real, data_unperturbed["data"].imag, axis=1))

    dists_learned = cdist(np.append(prediction[:, 0], prediction[:, 1], axis=1), np.append(
        data_unperturbed["data"].real, data_unperturbed["data"].imag, axis=1))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig = plt.figure(figsize=(POINTS_W/72, 2*0.33*POINTS_W/72))
    ax1 = fig.add_subplot(221)
    pl1 = ax1.pcolor(np.linspace(-1, 1, data_unperturbed["N"]), data_unperturbed["tt"][::10],
                     data_unperturbed["data"][:-1:10].real, rasterized=True)
    ax1.set_ylabel(r'$t$', labelpad=0)
    ax1.set_xlabel(r'$\phi_1$', labelpad=0)
    cbar1 = plt.colorbar(pl1)
    cbar1.set_label('Re $W$', labelpad=-6)
    ax2 = fig.add_subplot(222)
    pl2 = ax2.pcolor(np.linspace(-1, 1, data_unperturbed["N"]), data_unperturbed["tt"][::10],
                     prediction[::10, 0], rasterized=True)
    phi_arr = np.linspace(-1, 1, data_unperturbed["N"])
    ax2.axvline(x=(phi_arr[3]+phi_arr[4])/2, ymin=0, ymax=1, color='white', lw=1)
    ax2.axvline(x=(phi_arr[-4]+phi_arr[-5])/2, ymin=0, ymax=1, color='white', lw=1)
    ax2.set_ylabel(r'$t$', labelpad=0)
    ax2.set_xlabel(r'$\phi_1$', labelpad=0)
    cbar2 = plt.colorbar(pl2)
    cbar2.set_label('Re $W$', labelpad=-6)
    ax3 = fig.add_subplot(223)
    ax3.plot(data_unperturbed["tt"][:], np.min(dists_neg, axis=1)[:-1], label='$d$ true')
    ax3.plot(data_unperturbed["tt"][:], np.min(dists_learned, axis=1), '--', label='$d$ learned')
    ax3.set_xlabel(r'$t$', labelpad=2)
    ax3.set_ylabel(r'$d$', labelpad=0)
    plt.legend()
    ax4 = fig.add_subplot(224)
    ax4.plot(data_unperturbed["omega"], data_unperturbed["phi"])
    ax4.set_xlabel(r'$\omega$', labelpad=0)
    ax4.set_ylabel(r'$\phi_1$', labelpad=0)
    plt.subplots_adjust(top=0.96, wspace=0.3, right=0.95, bottom=0.1, left=0.1, hspace=0.3)
    ax1.text(-0.28, 1., r'$\mathbf{a}$', transform=ax1.transAxes, weight='bold',
             fontsize=plt.rcParams['axes.titlesize'], fontweight=plt.rcParams['axes.titleweight'])
    ax2.text(-0.28,  1., r'$\mathbf{b}$', transform=ax2.transAxes, weight='bold', fontsize=12)
    ax3.text(-0.22, 1., r'$\mathbf{c}$', transform=ax3.transAxes, weight='bold', fontsize=12)
    ax4.text(-0.22,  1., r'$\mathbf{d}$', transform=ax4.transAxes, weight='bold', fontsize=12)
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
            integrate_system_gamma(config["SYSTEM"], int(config["TRAINING"]["n_train"]) +
                                   int(config["TRAINING"]["n_test"]),
                                   config["GENERAL"]["save_dir"]+'/dat/',
                                   verbose=verbose)
        else:
            integrate_system(config["SYSTEM"], int(config["TRAINING"]["n_train"]) +
                             int(config["TRAINING"]["n_test"]),
                             config["GENERAL"]["save_dir"]+'/dat/',
                             verbose=verbose)

    # Create Dataset
    dataset_train = utils.Dataset(0, int(config["TRAINING"]["n_train"]), config["MODEL"],
                                  path=config["GENERAL"]["save_dir"], verbose=verbose)
    dataset_test = utils.Dataset(int(config["TRAINING"]["n_train"]),
                                 int(config["TRAINING"]["n_train"]) +
                                 int(config["TRAINING"]["n_test"]),
                                 config["MODEL"],
                                 path=config["GENERAL"]["save_dir"], verbose=verbose)

    if config["GENERAL"].getboolean("use_dmaps"):
        utils.dmaps_transform(int(config["TRAINING"]["n_train"]) +
                              int(config["TRAINING"]["n_test"]), dataset_train,
                              path=config["GENERAL"]["save_dir"], verbose=verbose)
        dataset_train = utils.Dataset(0, int(config["TRAINING"]["n_train"]), config["MODEL"],
                                      path=config["GENERAL"]["save_dir"], verbose=verbose)
        dataset_test = utils.Dataset(int(config["TRAINING"]["n_train"]),
                                     int(config["TRAINING"]["n_train"]) +
                                     int(config["TRAINING"]["n_test"]),
                                     config["MODEL"],
                                     path=config["GENERAL"]["save_dir"], verbose=verbose)

    dataloader_train = utils.FastDataLoader(
        dataset_train, batch_size=int(config["TRAINING"]['batch_size']), shuffle=True,
        num_workers=int(config["TRAINING"]['num_workers']), pin_memory=True)
    dataloader_test = utils.FastDataLoader(
        dataset_test, batch_size=int(config["TRAINING"]['batch_size']), shuffle=False,
        num_workers=int(config["TRAINING"]['num_workers']), pin_memory=True)

    network = lpde.network.Network(config["MODEL"], n_vars=2)

    delta_x = 2*float(config["SYSTEM"]["gamma"])/int(config["SYSTEM"]["N"]) * \
        float(config["MODEL"]["rescale_dx"])

    if verbose:
        tests.test_fd_coeffs(network, path=config["GENERAL"]["save_dir"])
        tests.test_derivs(network, torch.tensor(dataset_train.x_data[:1],
                                                dtype=torch.get_default_dtype()),
                          torch.tensor([delta_x], dtype=torch.get_default_dtype()),
                          path=config["GENERAL"]["save_dir"])

    model = lpde.model.Model(dataloader_train, dataloader_test, network, config["TRAINING"],
                             path=config["GENERAL"]["save_dir"]+'/')

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
        model = lpde.model.Model(dataloader_train, dataloader_test, network, config["TRAINING"],
                                 path=config["GENERAL"]["save_dir"]+'/')
        model.load_network('test.model')
        tests.test_learned_dt(model, dataset_test, mint.f,
                              path=config["GENERAL"]["save_dir"], idx=0)
        tests.test_transient_dynamics(model, dataset_test, dataset_train.svd,
                                      idx=int(config["TRAINING"]["n_train"]), t_off=0,
                                      path=config["GENERAL"]["save_dir"])


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config/config.cfg')
    main(config)

    make_plot_paper(config)
