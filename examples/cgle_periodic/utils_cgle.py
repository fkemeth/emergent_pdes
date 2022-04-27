import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pickle
import findiff
# from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
from scipy.integrate import ode
from scipy import interpolate

import fun.dmaps as dmaps

torch.set_default_dtype(torch.float32)


def get_period(a):
    """Calculate local minimal and return interval between first and second."""
    minima = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True] &
                      np.r_[True, a[1:] < 1e-2])[0]
    return minima[1]-minima[0]


def reset_data(config):
    print("Resetting...")
    os.remove(config["GENERAL"]["save_dir"]+'/v_scaled.npy')
    p_list = [-1, 0, 1]
    for i in range(int(config["TRAINING"]["n_train"]) +
                   int(config["TRAINING"]["n_test"])):
        for p in p_list:
            pkl_file = open(config["GENERAL"]["save_dir"] + '/dat/run' +
                            str(i) + '_p_'+str(p)+'.pkl', 'rb')
            data_dict = pickle.load(pkl_file)
            pkl_file.close()

            data_dict["data"] = data_dict["data_org"]
            del data_dict["data_org"]

            output = open(config["GENERAL"]["save_dir"] + '/dat/run' +
                          str(i) + '_p_'+str(p)+'.pkl', 'wb')
            pickle.dump(data_dict, output)
            output.close()


class Dataset(torch.utils.data.Dataset):
    """ Simple dataloader for Matthews simulation data."""

    def __init__(self, start_idx, end_idx, config, path, verbose=False, include_0=False):
        # Load data
        self.save_dir = path+'/dat/'
        # Length of left and right part for which no dt information is available
        self.off_set = int((int(config["kernel_size"])-1)/2)
        self.use_fd_dt = config.getboolean("use_fd_dt")
        self.rescale_dx = float(config["rescale_dx"])
        self.use_param = config.getboolean("use_param")
        self.verbose = verbose
        self.include_0 = include_0
        self.x_data, self.delta_x, self.y_data, self.param = self.load_data(start_idx, end_idx)
        self.n_samples = self.x_data.shape[0]

        self.svd = TruncatedSVD(n_components=int(config["svd_modes"]), n_iter=42, random_state=42)
        # self.svd.fit(self.x_data.reshape(self.n_samples, -1))
        self.svd.fit(self.x_data[::10].reshape(int(self.n_samples/10), -1))

        print('SVD variance explained: '+str(self.svd.explained_variance_ratio_.sum()))

    def load_data(self, start_idx, end_idx):
        """ Load and prepare data."""
        x_data = []
        delta_x = []
        param = []
        for idx in range(start_idx, end_idx):
            if self.include_0:
                p_list = [0, -1, 1]
            else:
                p_list = [-1, 1]
            for p in p_list:
                pkl_file = open(self.save_dir+'/run'+str(idx) + '_p_'+str(p)+'.pkl', 'rb')
                data = pickle.load(pkl_file)
                pkl_file.close()
                x_data.append(data["data"])
                delta_x.append(np.repeat(data["L"]/data["N"], len(data["data"])))
                param.append(np.repeat(data["param"], len(data["data"])))

        # Delta t for temporal finite difference estimation
        self.delta_t = (data["tmax"]-data["tmin"])/data["T"]
        if self.verbose:
            print('Using delta_t of '+str(self.delta_t))

        # Prepare data
        y_data = []
        for idx, data_point in enumerate(x_data):
            if self.use_fd_dt:
                y_data.append((data_point[1:, self.off_set:-self.off_set] -
                               data_point[:-1, self.off_set:-self.off_set])/self.delta_t)
            # If fd is attached to model, remove off set. TODO do fd here.
            x_data[idx] = x_data[idx][:-1]
            delta_x[idx] = delta_x[idx][:-1]
            param[idx] = param[idx][:-1]

        x_data = np.stack((np.concatenate(x_data).real,
                           np.concatenate(x_data).imag), axis=-1)
        y_data = np.stack((np.concatenate(y_data).real,
                           np.concatenate(y_data).imag), axis=-1)
        delta_x = np.concatenate(delta_x, axis=0)*self.rescale_dx
        param = (np.concatenate(param, axis=0) - 1.75)/0.02
        return np.transpose(x_data, (0, 2, 1)), delta_x, np.transpose(y_data, (0, 2, 1)), param

    def get_data(self, split_boundaries=False):
        """Return prepared data."""
        if split_boundaries:
            left_bound = self.x_data[:, :, :self.off_set]
            right_bound = self.x_data[:, :, -self.off_set:]
            return left_bound, self.x_data[:, :, self.off_set:-self.off_set], right_bound, \
                self.delta_x, self.y_data, self.param
        return self.x_data, self.delta_x, self.y_data, self.param

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        _x = torch.tensor(self.x_data[index], dtype=torch.get_default_dtype())
        _dx = torch.tensor(self.delta_x[index], dtype=torch.get_default_dtype())
        _y = torch.tensor(self.y_data[index], dtype=torch.get_default_dtype())
        _p = torch.tensor(self.param[index], dtype=torch.get_default_dtype())
        return _x, _dx, _y, _p


def dmaps_transform(n_total, dataset, path, verbose=False):
    """Parametrize data using first diffusion mode."""
    p_list = [0, -1, 1]
    try:
        v_scaled = np.load(path+'/v_scaled.npy')
    except:
        if 'sle_dmaps' in path:
            _, evecs = dmaps.dmaps(np.transpose(dataset.x_data[:, 0]+1.0j*dataset.x_data[:, 1]),
                                   eps=2e1, alpha=1)
        elif 'sle_gamma' in path:
            _, evecs = dmaps.dmaps(np.transpose(dataset.x_data[:, 0]+1.0j*dataset.x_data[:, 1]),
                                   eps=1e1, alpha=1)
        elif 'cgle' in path:
            _, evecs = dmaps.dmaps(np.transpose(dataset.x_data[:, 0]+1.0j*dataset.x_data[:, 1]),
                                   alpha=1)
        else:
            raise ValueError("System dmaps not implemented yet.")

        v_scaled = 2*(evecs[:, 1]-np.min(evecs[:, 1])) / \
            (np.max(evecs[:, 1])-np.min(evecs[:, 1]))-1.
        if v_scaled[-1] < 0:
            v_scaled = -v_scaled
        for data_idx in range(n_total):
            print(data_idx)
            for p in p_list:
                pkl_file = open(path + '/dat/run'+str(data_idx) + '_p_'+str(p)+'.pkl', 'rb')
                data_dict = pickle.load(pkl_file)
                pkl_file.close()
                data_dict["phi"] = v_scaled
                phi_arr = np.linspace(-1, 1, int(data_dict["N"]))
                tt_arr = np.linspace(data_dict["tmin"], data_dict["tmax"], data_dict["T"]+1)
                if "data_org" not in data_dict.keys():
                    data_dict["data_org"] = data_dict["data"].copy()
                # print(tt_arr.shape)
                # print(data_dict["phi"].shape)
                # print(data_dict["data_org"].shape)
                # data_dict["data"] = np.zeros(data_dict["data_org"].shape, dtype='complex')
                # tt_arr, phi_arr = np.meshgrid(tt_arr, phi_arr)
                # for i in range(data_dict["data_org"].shape[0]):
                #     fit_phi_r = interpolate.Rbf(v_scaled,
                #                                 data_dict["data_org"][i].real,
                #                                 epsilon=2)
                #     fit_phi_i = interpolate.Rbf(v_scaled,
                #                                 data_dict["data_org"][i].imag,
                #                                 epsilon=2)
                #     data_dict["data"][i] = fit_phi_r(
                #         phi_arr) + 1.0j*fit_phi_i(phi_arr)
                fit_phi_r = interpolate.interp2d(data_dict["phi"], tt_arr,
                                                 data_dict["data_org"].real, kind='cubic')
                fit_phi_i = interpolate.interp2d(data_dict["phi"], tt_arr,
                                                 data_dict["data_org"].imag, kind='cubic')
                data_dict["data"] = fit_phi_r(
                    phi_arr, tt_arr) + 1.0j*fit_phi_i(phi_arr, tt_arr)
                # fit_phi_r = interpolate.RectBivariateSpline(tt_arr, data_dict["phi"],
                #                                             data_dict["data_org"].real, s=5)
                # fit_phi_i = interpolate.RectBivariateSpline(tt_arr, data_dict["phi"],
                #                                             data_dict["data_org"].imag, s=5)
                # data_dict["data"] = fit_phi_r(
                #     tt_arr, phi_arr) + 1.0j*fit_phi_i(tt_arr, phi_arr)

                data_dict["dmaps_fit_r"] = fit_phi_r
                data_dict["dmaps_fit_i"] = fit_phi_i
                output = open(path + '/dat/run' + str(data_idx) + '_p_'+str(p)+'.pkl', 'wb')
                pickle.dump(data_dict, output)
                output.close()
        print('Saveing v_scaled.')
        np.save(path+'/v_scaled.npy', v_scaled)
        if verbose:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(data_dict["xx"], v_scaled)
            ax.set_xlabel('$x$')
            ax.set_ylabel(r'$\phi_1$')
            plt.savefig(path+'/tests/v_scaled.pdf')
            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.pcolor(data_dict["data"][::10].real)
            ax.set_xlabel('$i$')
            ax.set_ylabel('$t$')
            plt.savefig(path+'/tests/data_transformed.pdf')
            plt.show()
