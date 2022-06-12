"""
Run Matthews example using specified config file.
"""
import os
import pickle
import shutil
import configparser
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import tqdm
import torch
import lpde
from torch.utils.tensorboard import SummaryWriter

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

    dataset = utils.Dataset(int(config["TRAINING"]["n_train"]) +
                            int(config["TRAINING"]["n_test"])-1,
                            int(config["TRAINING"]["n_train"]) +
                            int(config["TRAINING"]["n_test"]),
                            config["MODEL"],
                            path=config["GENERAL"]["save_dir"])

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=int(config["TRAINING"]['batch_size']), shuffle=True,
        num_workers=int(config["TRAINING"]['num_workers']), pin_memory=True)

    network = lpde.network.Network(config["MODEL"], n_vars=2)

    model = lpde.model.Model(dataloader, dataloader, network, config["TRAINING"])
    model.load_network('test.model')

    limit_amps_true = []
    limit_amps_learned = []
    gamma_list = []
    for i in range(int(config["TRAINING"]["n_train"])):
        print(i)
        dataset = utils.Dataset(i, i+1, config["MODEL"],
                                path=config["GENERAL"]["save_dir"])

        prediction = model.integrate_svd(dataset, dataset_train.svd, 0, 20000-1)
        if i == 0:
            prediction0 = model.integrate_svd(dataset, dataset_train.svd, 0, 20000-1)

        limit_amps_true.append(
            np.mean(np.abs(dataset.x_data[20000-1, 0]+1.0j*dataset.x_data[20000-1, 1])))
        limit_amps_learned.append(np.mean(np.abs(prediction[-1, 0]+1.0j*prediction[-1, 1])))
        gamma_list.append(dataset.param[0]*0.02+1.75)

    pkl_file = open(config["GENERAL"]["save_dir"]+'/dat/run' +
                    str(int(config["TRAINING"]["n_train"])+int(config["TRAINING"]["n_test"])-1) + '_p_'+str(-1)+'.pkl', 'rb')
    data_unperturbed = pickle.load(pkl_file)
    pkl_file.close()

    v_scaled = np.load(config["GENERAL"]["save_dir"]+'/v_scaled.npy')

    df_data = {'gamma': gamma_list,
               'limit_amps_true': limit_amps_true,
               'limit_amps_predicted': limit_amps_learned}
    df = pd.DataFrame(df_data)
    df.to_excel(r'Source_Data_Figure_4.xlsx',
                sheet_name='Figure 4', index=False)

    np.save('Source_Data_Figure_4_Inset_1.npy', prediction[::10, 0])
    np.save('Source_Data_Figure_4_Inset_2.npy', prediction0[::10, 0])

    fig = plt.figure(figsize=(POINTS_W/72, 0.66*POINTS_W/72))
    ax1 = fig.add_subplot(111)
    scat1 = ax1.scatter(gamma_list, limit_amps_true, label='true')
    scat2 = ax1.scatter(gamma_list, limit_amps_learned, label='learned',
                        marker='+')
    ax1.set_xlabel(r'$\gamma$', labelpad=-2)
    ax1.set_ylabel(r'$\langle | W_{\mbox{limit}}|\rangle$', labelpad=-1)
    ax1.set_ylim((-0.005, 0.18))
    ax1.set_xlim((min(gamma_list)-0.002, max(gamma_list)+0.002))
    axins1 = ax1.inset_axes([0.67, 0.33, 0.32, 0.42])
    axins1.pcolor(np.linspace(-1, 1, data_unperturbed["N"]), data_unperturbed["tt"][::10],
                  prediction[::10, 0], rasterized=True)
    phi_arr = np.linspace(-1, 1, data_unperturbed["N"])
    axins1.axvline(x=(phi_arr[3]+phi_arr[4])/2, ymin=0, ymax=1, color='white', lw=1)
    axins1.axvline(x=(phi_arr[-4]+phi_arr[-5])/2, ymin=0, ymax=1, color='white', lw=1)
    axins1.set_xlabel(r'$\phi_1$')
    axins1.set_ylabel(r'$t$', labelpad=-2)
    axins2 = ax1.inset_axes([0.24, 0.55, 0.32, 0.42])
    axins2.pcolor(np.linspace(-1, 1, data_unperturbed["N"]), data_unperturbed["tt"][::10],
                  prediction0[::10, 0], rasterized=True)
    axins2.axvline(x=(phi_arr[3]+phi_arr[4])/2, ymin=0, ymax=1, color='white', lw=1)
    axins2.axvline(x=(phi_arr[-4]+phi_arr[-5])/2, ymin=0, ymax=1, color='white', lw=1)
    axins2.set_xlabel(r'$\phi_1$')
    axins2.set_ylabel(r'$t$', labelpad=-2)
    plt.legend(fontsize=8)
    ax1.annotate("", xy=(1.7995, 0.001), xytext=(1.79, 0.05),
                 arrowprops=dict(arrowstyle="->"))
    ax1.annotate("", xy=(1.7005, 0.117), xytext=(1.719, 0.12),
                 arrowprops=dict(arrowstyle="->"))
    plt.subplots_adjust(top=0.94, wspace=0.45, right=0.98, bottom=0.1, hspace=0.3, left=0.15)
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
