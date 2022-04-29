import os
from tqdm.auto import tqdm
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

import dm as diffusion_maps
from fitting import BasisFit, BasisFitter


import int.twoConductanceEquations as twoConductanceEquations
from fitting import BasisFitter

torch.set_default_dtype(torch.float32)

config = {}
config["GENERAL"] = {}
config["GENERAL"]["verbose"] = True

config["DATA"] = {}
config["DATA"]["domain_size"] = 128
config["DATA"]["iapp_min"] = 15
config["DATA"]["iapp_max"] = 24
config["DATA"]["tmin"] = 100
config["DATA"]["tmax"] = 120
config["DATA"]["T"] = 10000
config["DATA"]["T_off"] = 400
config["DATA"]["n_train"] = 3
config["DATA"]["n_test"] = 1
config["DATA"]["eps"] = 1e-1
config["DATA"]["N_int"] = 1024
config["DATA"]["N"] = 64
config["DATA"]["p_list"] = [-1, 1]
config["DATA"]["path"] = 'data/'

config["MODEL"] = {}
config["MODEL"]["kernel_size"] = 5
config["MODEL"]["svd_modes"] = 10
config["MODEL"]["transform_data"] = True
config["MODEL"]['device'] = 'cuda'
config["MODEL"]['hypervisc'] = False
config["MODEL"]['n_filters'] = 64
config["MODEL"]['n_layers'] = 3
config["MODEL"]["n_derivs"] = 3
config["MODEL"]["rescale_dx"] = 1

config["TRAINING"] = {}
config["TRAINING"]['batch_size'] = 128
config["TRAINING"]['num_workers'] = 8
config["TRAINING"]["reduce_factor"] = .5
config["TRAINING"]["patience"] = 10
config["TRAINING"]["lr"] = 2e-3
config["TRAINING"]['epochs'] = 100
config["TRAINING"]['proceed_training'] = False
config["TRAINING"]['dtype'] = torch.float32

torch.set_default_dtype(config["TRAINING"]['dtype'])


def from_hV(data):
    # return (data.real+37)/15+1.0j*(data.imag-0.42)/0.1
    return np.stack(((data[:, :, 0]+37)/30, (data[:, :, 1]-0.42)/0.2), axis=-1)


def to_hV(data):
    # return data.real*15-37+1.0j*(data.imag*0.1+0.42)
    return np.stack((data[:, :, 0]*30-37, (data[:, :, 1]*0.2+0.42)), axis=-1)


def from_hV_dt(dt_data):
    # return (data.real+37)/15+1.0j*(data.imag-0.42)/0.1
    return np.stack((dt_data[:, :, 0]/30, dt_data[:, :, 1]/0.2), axis=-1)


def to_hV_dt(dt_data):
    # return (data.real+37)/15+1.0j*(data.imag-0.42)/0.1
    return np.stack((dt_data[:, :, 0]*30, dt_data[:, :, 1]*0.2), axis=-1)


def R(fineState, fitter):
    fineState = fineState.T
    # fineState = fineState.reshape(2, 1024)
    coeffs = [fitter.fit(variable).coeffs for variable in fineState]
    return np.hstack(coeffs)


class ChungLuAdjMat(np.ndarray):
    p = None
    r = None

    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)


def chung_lu(N, p=0.9, r=0.25, normalize=False, symmetric=True):
    """
    >>> A = chungLu(40, .5, .2)
    >>> from __init__ import matrixLikeAdjDict
    >>> matrixLikeAdjDict(A).shape
    (40, 40)
    >>> assert A.sum() != 0
    >>> from minseokepj.plotting import showMat
    >>> #showMat(A, show=True)
    >>> A = chungLu(128, .3, .6, symmetric=True)
    >>> assert (A == A.T).all()
    """
    P = np.empty((N, N))
    w = p * float(N) * (np.arange(1, N+1) / float(N)) ** r
    wsum = w.sum()
    done = {}
    for i in range(N):
        for j in range(N):
            if (i, j) not in done and (j, i) not in done:  # prevent excessive iteration
                P[i, j] = w[i] * w[j]
                P[j, i] = P[i, j]
                done[(i, j)] = True
    P /= wsum
    A = (P > np.random.random((N, N))).astype(int)
    if symmetric:
        for i in range(N):
            for j in range(N):
                A[i, j] = A[j, i]
        assert (A == A.T).all(), "A not symmetric."
    if normalize:
        A *= float(N ** 2) / A.sum()
    A = ChungLuAdjMat(A)
    A.p = p
    A.r = r
    return A


def integrate_system(config, n, path, verbose=False):
    """Integrate HH system."""

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

    for i in range(n):
        # for p in config["p_list"]:
        for p in [0, -1, 1]:
            sim_ip = twoConductanceEquations.TwoCondEqnSim(
                N=pars["N"], Iapp=pars["domain"], A=pars["A"])
            initial_condition = np.expand_dims(sim.history.y[:, int(i*config["T_off"])], axis=-1)
            initial_condition = np.stack(
                (initial_condition[:pars["N"]].T, initial_condition[pars["N"]:].T), axis=-1)
            perturbed_snapshot = to_hV((1+config["eps"]*p)*from_hV(initial_condition))
            perturbed_snapshot = np.hstack(
                (perturbed_snapshot[0, :, 0], perturbed_snapshot[0, :, 1]))
            sim_ip.integrate(X0=perturbed_snapshot, t_span=[0, tmax-tmin],
                             t_eval=np.linspace(0, tmax-tmin, T+1), rtol=1e-8, atol=1e-10)

            data_dict = {}
            data_dict["xx"] = pars["domain"]
            data_dict["tt"] = sim_ip.history.t
            data_dict["Iapp"] = pars["domain"]
            data_dict["N"] = pars["N"]
            data_dict["A"] = pars["A"]
            data_dict["data"] = np.stack(
                (sim_ip.history.y[:pars["N"]].T, sim_ip.history.y[pars["N"]:].T), axis=-1)
            data_dict["Ainit"] = data_dict["data"][0]

            output = open(path+str(i) + '_p_'+str(p)+'.pkl', 'wb')
            pickle.dump(data_dict, output)
            output.close()

    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data_dict["xx"], data_dict["data"][-1, :, 0], '.', label='h')
        ax.plot(data_dict["xx"], data_dict["data"][-1, :, 1], '.', label='V')
        ax.set_xlabel(r'$I$')
        plt.title('snapshot')
        plt.legend()
        plt.savefig(path+'snapshot.pdf')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data_dict["xx"], from_hV(data_dict["data"])[-1, :, 0], '.', label='u')
        ax.plot(data_dict["xx"], from_hV(data_dict["data"])[-1, :, 1], '.', label='v')
        ax.set_xlabel(r'$I$')
        plt.title('snapshot')
        plt.legend()
        plt.savefig(path+'snapshot_trans.pdf')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        pl1 = ax.pcolor(data_dict["xx"][np.argsort(data_dict["xx"])], data_dict["tt"][::40],
                        data_dict["data"][::40, np.argsort(data_dict["xx"]), 0],
                        rasterized=True)
        ax.set_xlabel(r'$I$')
        ax.set_ylabel(r'$t$')
        cbar = plt.colorbar(pl1, label='h')
        plt.savefig(path+'limit_cycle_pcolor.pdf')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.mean(data_dict["data"][:, :, 0], axis=1),
                np.mean(data_dict["data"][:, :, 1], axis=1))
        ax.set_xlabel(r'$h$')
        ax.set_xlabel(r'$V$')
        plt.savefig(path+'mean_dynamics.pdf')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.mean(from_hV(data_dict["data"])[:, :, 0], axis=1),
                np.mean(from_hV(data_dict["data"])[:, :, 1], axis=1))
        ax.set_xlabel(r'$h$')
        ax.set_xlabel(r'$V$')
        plt.savefig(path+'mean_dynamics_trans.pdf')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.mean(data_dict["data"][:, :, 0], axis=1),
                np.mean(data_dict["data"][:, :, 1], axis=1))
        ax.plot(np.mean(to_hV(from_hV(data_dict["data"]))[:, :, 0], axis=1),
                np.mean(to_hV(from_hV(data_dict["data"]))[:, :, 1], axis=1))
        ax.set_xlabel(r'$h$')
        ax.set_xlabel(r'$V$')
        plt.savefig(path+'mean_dynamics_trans_orig.pdf')
        plt.show()


def do_dmaps(config, n, path, eps=4000, verbose=True):
    data = []
    for i in range(np.min((n, 5))):
        # for p in [0, -1, 1]:
        pkl_file = open(path+str(i) + '_p_'+str(p)+'.pkl', 'rb')
        pkl_data = pickle.load(pkl_file)
        pkl_file.close()
        data.append(pkl_data["data"])
    data = np.concatenate(data)
    print(data.shape)
    data = from_hV(data)
    # D, V = dmaps.dmaps(data[::20, :, 0].T+1.0j*data[::20, :, 1].T, alpha=1, eps=4000)
    dmap = diffusion_maps.SparseDiffusionMaps(points=np.vstack((data[::200, :, 0],
                                                                data[::200, :, 1])).T,
                                              epsilon=eps,
                                              num_eigenpairs=12,
                                              cut_off=np.inf,
                                              renormalization=0,
                                              normalize_kernel=True,
                                              use_cuda=False)
    V = dmap.eigenvectors.T
    D = dmap.eigenvalues.T

    # U, s, V =
    # D, V = dmaps.dmaps(data[::20, :, 0].T+1.0j*data[::20, :, 1].T, alpha=1)
    V[:, 1] = 2*(V[:, 1]-np.min(V[:, 1])) / \
        (np.max(V[:, 1])-np.min(V[:, 1]))-1.
    V[:, 2] = 2*(V[:, 2]-np.min(V[:, 2])) / \
        (np.max(V[:, 2])-np.min(V[:, 2]))-1.

    if np.mean(V[:, 1]) < 0:
        V[:, 1] = -V[:, 1]
    if np.mean(V[:, 2]) < 0:
        V[:, 2] = -V[:, 2]

    np.save(path+'dmaps_D.npy', D)
    np.save(path+'dmaps_V.npy', V)

    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sc = ax.scatter(V[:, 1], V[:, 2], c=pkl_data["xx"])
        ax.set_xlabel(r'$\phi_1$')
        ax.set_ylabel(r'$\phi_2$')
        cbar = plt.colorbar(sc, label=r'$I_{app}^i$')
        plt.savefig(path+'phi1phi2_iappi.pdf')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        sc = ax.scatter(V[:, 1], V[:, 2])
        ax.set_xlabel(r'$\phi_1$')
        ax.set_ylabel(r'$\phi_2$')
        plt.savefig(path+'phi1phi.pdf')
        plt.savefig(path+'phi1phi.png', dpi=300)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(V[:, 1], V[:, 2], V[:, 3])
        ax.set_xlabel(r'$\phi_1$')
        ax.set_ylabel(r'$\phi_2$')
        ax.set_zlabel(r'$\phi_3$')
        plt.savefig(path+'phi1phi2phi3.pdf')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        sc = ax.scatter(V[:, 1], V[:, 2], c=np.sum(pkl_data["A"], axis=1))
        ax.set_xlabel(r'$\phi_1$')
        ax.set_ylabel(r'$\phi_2$')
        cbar = plt.colorbar(sc, label=r'$\kappa$')
        plt.savefig(path+'phi1phi2_kappa.pdf')
        plt.show()


def dmaps_transform_old(config, n, path, verbose=True):
    D = np.load(path+'dmaps_D.npy')
    V = np.load(path+'dmaps_V.npy')

    v_scaled1 = V[:, 1]
    v_scaled2 = V[:, 2]

    ph1abs = 0.6
    ph2min = 0.25

    phihat1 = np.linspace(-ph1abs, ph1abs, config["N"])
    phihat2 = np.linspace(ph2min, 1, config["N"])

    phihat1, phihat2 = np.meshgrid(phihat1, phihat2)

    for i in range(n):
        print(i)
        # for p in config["p_list"]:
        for p in [0, -1, 1]:
            print(p)
            pkl_file = open(path+str(i) + '_p_'+str(p)+'.pkl', 'rb')
            pkl_data = pickle.load(pkl_file)
            pkl_file.close()
            data = from_hV(pkl_data["data"])
            pkl_data["data_fitted"] = np.zeros((data.shape[0], config["N"], config["N"], 2))
            progress_bar = tqdm(range(data.shape[0]), leave=True, position=0)
            for k in progress_bar:
                fit_phi_r = interpolate.Rbf(v_scaled1[np.all((np.abs(v_scaled1) < ph1abs,
                                                              v_scaled2 > ph2min), axis=0)],
                                            v_scaled2[np.all((np.abs(v_scaled1) < ph1abs,
                                                              v_scaled2 > ph2min), axis=0)],
                                            data[k, np.all((np.abs(v_scaled1) < ph1abs,
                                                            v_scaled2 > ph2min),
                                                           axis=0), 0], smooth=2)
                fit_phi_i = interpolate.Rbf(v_scaled1[np.all((np.abs(v_scaled1) < ph1abs,
                                                              v_scaled2 > ph2min), axis=0)],
                                            v_scaled2[np.all((np.abs(v_scaled1) < ph1abs,
                                                              v_scaled2 > ph2min), axis=0)],
                                            data[k,
                                                 np.all((np.abs(v_scaled1) < ph1abs,
                                                         v_scaled2 > ph2min),
                                                        axis=0), 1], smooth=2)
                pkl_data["data_fitted"][k, :, :, 0] = fit_phi_r(phihat1, phihat2)
                pkl_data["data_fitted"][k, :, :, 1] = fit_phi_i(phihat1, phihat2)
            output = open(path+str(i) + '_p_'+str(p)+'.pkl', 'wb')
            pickle.dump(pkl_data, output)
            output.close()

    if verbose:

        xi, yi = np.meshgrid(np.linspace(-ph1abs, ph1abs,
                                         config["N"]+1), np.linspace(ph2min, 1, config["N"]+1))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
        ax.plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
        scat1 = ax.scatter(v_scaled1, v_scaled2, s=1, c=pkl_data["xx"])
        ax.set_xlabel(r'$\phi_1$')
        ax.set_ylabel(r'$\phi_2$')
        cbar1 = plt.colorbar(scat1)
        cbar1.set_label(r'$I_{app}^i$')
        plt.savefig(path+'/dmaps_transform_2d_grid_N_'+str(config["N"])+'.pdf')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(v_scaled1[np.all((np.abs(v_scaled1) < ph1abs,
                                     v_scaled2 > ph2min), axis=0)],
                   v_scaled2[np.all((np.abs(v_scaled1) < ph1abs,
                                     v_scaled2 > ph2min), axis=0)],
                   data[k, np.all((np.abs(v_scaled1) < ph1abs,
                                   v_scaled2 > ph2min),
                                  axis=0), 0], zorder=10, color='black')
        ax.plot_wireframe(phihat1, phihat2,
                          pkl_data["data_fitted"][k, :, :, 0], rstride=2, cstride=2)
        ax.set_xlabel(r'$\phi_1$')
        ax.set_ylabel(r'$\phi_2$')
        ax.set_zlabel(r'h')
        plt.savefig(path+'/dmaps_transform_data_3d_N_'+str(config["N"])+'.pdf')
        plt.show()


def dmaps_transform(config, n, path, verbose=True):
    D = np.load(path+'dmaps_D.npy')
    V = np.load(path+'dmaps_V.npy')

    v_scaled1 = V[:, 1]
    v_scaled2 = V[:, 2]

    Xfull, Yfull = np.meshgrid(np.linspace(-1, 1, config["N"]), np.linspace(0, 1, config["N"]))

    maxOrd = 2
    hets = np.vstack((v_scaled1, v_scaled2)).T
    fitter = BasisFitter(
        hets, maxOrd=maxOrd,
        basis=None
    )

    for i in range(n):
        print(i)
        # for p in config["p_list"]:
        for p in [0, -1, 1]:
            print(p)
            pkl_file = open(path+str(i) + '_p_'+str(p)+'.pkl', 'rb')
            pkl_data = pickle.load(pkl_file)
            pkl_file.close()
            data = from_hV(pkl_data["data"])
            pkl_data["data_fitted"] = np.zeros((data.shape[0], config["N"], config["N"], 2))
            progress_bar = tqdm(range(data.shape[0]), leave=True, position=0)
            for k in progress_bar:
                coeffs = R(from_hV(pkl_data["data"][k:k+1, :, :])[0], fitter)
                # coeffs = coeffs.reshape(int(coeffs.size/2), 2).T
                coeffs = coeffs.reshape(2,  int(coeffs.size/2))
                fit0 = BasisFit(coeffs[0, :], fitter.basis)
                fit1 = BasisFit(coeffs[1, :], fitter.basis)

                pkl_data["data_fitted"][k, :, :, 0] = fit0(Xfull, Yfull)
                pkl_data["data_fitted"][k, :, :, 1] = fit1(Xfull, Yfull)
            output = open(path+str(i) + '_p_'+str(p)+'.pkl', 'wb')
            pickle.dump(pkl_data, output)
            output.close()

    if verbose:

        xi, yi = np.meshgrid(np.linspace(-1, 1,
                                         config["N"]+1), np.linspace(0, 1, config["N"]+1))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
        ax.plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
        scat1 = ax.scatter(v_scaled1, v_scaled2, s=1, c=pkl_data["xx"])
        ax.set_xlabel(r'$\phi_1$')
        ax.set_ylabel(r'$\phi_2$')
        cbar1 = plt.colorbar(scat1)
        cbar1.set_label(r'$I_{app}^i$')
        plt.savefig(path+'/dmaps_transform_2d_grid_N_'+str(config["N"])+'.pdf')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(v_scaled1, v_scaled2, data[k, :, 0], zorder=10, color='black')
        ax.plot_wireframe(Xfull, Yfull,
                          pkl_data["data_fitted"][k, :, :, 0], rstride=2, cstride=2)
        ax.set_xlabel(r'$\phi_1$')
        ax.set_ylabel(r'$\phi_2$')
        ax.set_zlabel(r'h')
        plt.savefig(path+'/dmaps_transform_data_3d_N_'+str(config["N"])+'.pdf')
        plt.show()


# if __name__ == "__main__":
#     integrate_system(config["DATA"], 1, path=config["DATA"]['path'], verbose=True)


class Dataset(torch.utils.data.Dataset):
    """ Simple dataloader for Matthews simulation data."""

    def __init__(self, start_idx, end_idx, config_data, config_model, path, use_svd=True):
        # Load data
        self.path = path
        # Length of left and right part for which no dt information is available
        self.off_set = int((int(config_model["kernel_size"])-1)/2)
        self.rescale_dx = float(config_model["rescale_dx"])
        self.p_list = config_data["p_list"]
        self.x_data, self.delta_x, self.delta_y, self.y_data = self.load_data(start_idx, end_idx)
        self.n_samples = self.x_data.shape[0]
        self.use_svd = use_svd
        if self.use_svd:
            self.svd = TruncatedSVD(n_components=int(
                config_model["svd_modes"]), n_iter=42, random_state=42)
            self.svd.fit(self.x_data.reshape(self.n_samples, -1))
            print('SVD variance explained: '+str(self.svd.explained_variance_ratio_.sum()))
        else:
            self.svd = None

    def load_data(self, start_idx, end_idx):
        """ Load and prepare data."""
        x_data = []
        delta_x = []
        delta_y = []
        for idx in range(start_idx, end_idx):
            for p in self.p_list:
                pkl_file = open(self.path+str(idx) + '_p_'+str(p)+'.pkl', 'rb')
                data = pickle.load(pkl_file)
                pkl_file.close()
                x_data.append(data["data_fitted"])
                delta_x.append(np.repeat(2./data["data"].shape[1], len(data["data"])))
                delta_y.append(np.repeat(1./data["data"].shape[2], len(data["data"])))

        # Delta t for temporal finite difference estimation
        self.delta_t = data["tt"][1]-data["tt"][0]
        print('Using delta_t of '+str(self.delta_t))

        # Prepare data
        y_data = []
        for idx, data_point in enumerate(x_data):
            y_data.append((data_point[1:, self.off_set:-self.off_set, self.off_set:-self.off_set] -
                           data_point[:-1, self.off_set:-self.off_set, self.off_set:-self.off_set])/self.delta_t)
            # If fd is attached to model, remove off set. TODO do fd here.
            x_data[idx] = x_data[idx][:-1]
            delta_x[idx] = delta_x[idx][:-1]
            delta_y[idx] = delta_y[idx][:-1]

        x_data = np.transpose(np.concatenate(x_data), (0, 3, 1, 2))
        print(x_data.shape)
        y_data = np.transpose(np.concatenate(y_data), (0, 3, 1, 2))
        delta_x = np.concatenate(delta_x, axis=0)*self.rescale_dx
        delta_y = np.concatenate(delta_y, axis=0)*self.rescale_dx
        return x_data, delta_x, delta_y, y_data

    def get_data(self, split_boundaries=False):
        """Return prepared data."""
        if split_boundaries:
            top_bound = self.x_data[:, :, :self.off_set]
            left_bound = self.x_data[:, :, :, :self.off_set]
            bot_bound = self.x_data[:, :, -self.off_set:]
            right_bound = self.x_data[:, :, :, -self.off_set:]
            return top_bound, left_bound, \
                self.x_data[:, :, self.off_set:-self.off_set, self.off_set:-self.off_set], \
                bot_bound, right_bound, \
                self.delta_x, self.delta_y, self.y_data
        return self.x_data, self.delta_x, self.delta_y, self.y_data

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        _x = torch.tensor(self.x_data[index], dtype=torch.get_default_dtype())
        _dx = torch.tensor(self.delta_x[index], dtype=torch.get_default_dtype())
        _dy = torch.tensor(self.delta_y[index], dtype=torch.get_default_dtype())
        _y = torch.tensor(self.y_data[index], dtype=torch.get_default_dtype())
        return _x, _dx, _dy, _y


class Network(nn.Module):
    """Pytorch network architecture."""

    def __init__(self, config):
        super(Network, self).__init__()
        self.kernel_size = int(config["kernel_size"])
        self.n_derivs = int(config["n_derivs"])
        self.device = config['device']
        self.register_buffer('coeffs', self.get_coeffs(max_deriv=self.n_derivs))

        num_features = int(4*(self.n_derivs+1))
        n_channels = int(config["n_filters"])
        layers = []
        for _ in range(int(config["n_layers"])):
            layers.append(nn.Conv1d(num_features, n_channels, [1], stride=1, padding=0, bias=True))
            layers.append(nn.Tanh())
            num_features = n_channels

        layers.append(nn.Conv1d(num_features, 2, [1], stride=1, padding=0, bias=True))

        self.network = nn.Sequential(*layers)
        self.trainable_parameters = sum(p.numel()
                                        for p in self.network.parameters() if p.requires_grad)

    def get_coeffs(self, min_deriv=0, max_deriv=5):
        """Get finite difference coefficients."""
        assert max_deriv > min_deriv, "Derivative range not specified"
        assert min_deriv >= 0, "Derivatives should be larger zero"

        coeffs = np.zeros((2*(max_deriv-min_deriv+1), 1, self.kernel_size, self.kernel_size))
        # Finite difference coefficients
        for i in range(min_deriv, max_deriv+1):
            # Get coefficient for certain derivative with maximal acc order for given kernel_size
            fd_coeff = np.array([1.])
            if i > 0:
                acc_order = 0
                while len(fd_coeff) < self.kernel_size:
                    acc_order += 2
                    fd_coeff = findiff.coefficients(i, acc_order)["center"]["coefficients"]
                assert len(fd_coeff) == self.kernel_size, \
                    "Finite difference coefficients do not match kernel"
                coeffs[2*i, 0, :, int((self.kernel_size-1)/2)] = fd_coeff  # x direction
                coeffs[2*i+1, 0, int((self.kernel_size-1)/2), :] = fd_coeff  # y direction
            else:
                coeffs[i, 0, int((self.kernel_size-1)/2), int((self.kernel_size-1)/2)] = 1.0
        return torch.tensor(coeffs, requires_grad=False,
                            dtype=torch.get_default_dtype()).to(self.device)

    def calc_derivs(self, x, dx, dy):
        """Calculate derivativers of input x."""
        finite_diffs = torch.cat([F.conv2d(x[:, :1], self.coeffs),
                                  F.conv2d(x[:, 1:], self.coeffs)], dim=1).to(self.device)
        x_scales = torch.cat([torch.pow(dx.unsqueeze(1), i)
                              for i in range(int(self.coeffs.shape[0]/2))], axis=-1)
        y_scales = torch.cat([torch.pow(dy.unsqueeze(1), i)
                              for i in range(int(self.coeffs.shape[0]/2))], axis=-1)
        x_scales = torch.cat((x_scales, x_scales), axis=1).unsqueeze(2).unsqueeze(2).repeat(1, 1,
                                                                                            finite_diffs.shape[-2],
                                                                                            finite_diffs.shape[-1]
                                                                                            ).to(self.device)
        y_scales = torch.cat((y_scales, y_scales), axis=1).unsqueeze(2).unsqueeze(2).repeat(1, 1,
                                                                                            finite_diffs.shape[-2],
                                                                                            finite_diffs.shape[-1]
                                                                                            ).to(self.device)
        return torch.cat([finite_diffs[:, ::2]/x_scales, finite_diffs[:, 1::2]/y_scales], axis=1)

    def forward(self, x, dx, dy):
        """Predict temporal derivative of input x."""
        # Calculate derivatives
        x = self.calc_derivs(x, dx, dy)
        x = torch.flatten(x, start_dim=2)
        # Forward through distributed parameter stack
        x = self.network(x)
        return x


class Model:
    def __init__(self, dataloader_train, dataloader_val, network, config, path):
        super().__init__()
        self.base_path = path

        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        self.net = network
        self.device = self.net.device
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")
        print('Using:', self.device)
        self.net = self.net.to(self.device)
        # self.net.device = self.device

        self.learning_rate = float(config["lr"])

        self.criterion = nn.MSELoss(reduction='sum').to(self.device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=int(config["patience"]),
            factor=float(config["reduce_factor"]), min_lr=1e-7)

    def train(self):
        """
        Train model over one epoch.

        Returns
        -------
        avg_loss: float
            Loss averaged over the training data
        """
        self.net = self.net.train()

        sum_loss, cnt = 0, 0
        for (data, delta_x, delta_y, target) in self.dataloader_train:
            data = data.to(self.device)
            delta_x = delta_x.to(self.device)
            delta_y = delta_y.to(self.device)
            target = torch.flatten(target, start_dim=2).to(self.device)

            # backward
            self.optimizer.zero_grad()

            # forward
            output = self.net(data, delta_x, delta_y)

            # compute loss
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            # measure accuracy on batch
            sum_loss += loss
            cnt += 1

        return sum_loss / cnt

    def validate(self):
        """
        Validate model on validation set.

        Updates learning rate using scheduler.

        Updates best accuracy.

        Returns
        -------
        avg_loss: float
            Loss averaged over the validation data
        """
        self.net = self.net.eval()

        sum_loss, cnt = 0, 0
        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(self.dataloader_val):
            for (data, delta_x, delta_y, target) in self.dataloader_val:
                data = data.to(self.device)
                delta_x = delta_x.to(self.device)
                delta_y = delta_y.to(self.device)
                target = torch.flatten(target, start_dim=2).to(self.device)

                # forward
                output = self.net(data, delta_x, delta_y)

                # loss / accuracy
                sum_loss += self.criterion(output, target)
                cnt += 1

        # Learning Rate reduction
        self.scheduler.step(sum_loss / cnt)

        return sum_loss / cnt

    def save_network(self, name):
        """
        Save model to disk.

        Arguments
        -------
        name: str
            Model filename.

        Returns
        -------
        name: str
            Model filename.
        """
        model_file_name = self.base_path+name
        torch.save(self.net.state_dict(), model_file_name)
        return name

    def load_network(self, name):
        """
        Load model from disk.

        Arguments
        -------
        name: str
            Model filename.
        """
        model_file_name = self.base_path+name
        self.net.load_state_dict(torch.load(model_file_name))

    def integrate(self, dataset, svd, idx, horizon):
        """Integrate idx'th snapshot of dataset for horizon time steps."""
        # assert idx == 0, "Only idx = 0 implemented."
        top_bound, left_bound, _, bot_bound, right_bound, _, _, _ = dataset.get_data(True)
        data0 = svd.inverse_transform(
            svd.transform(dataset.x_data[idx].reshape(1, -1)))
        data = []
        data.append(data0.reshape(2, int(np.sqrt(len(data0[0])/2)),
                                  int(np.sqrt(len(data0[0])/2))))
        for i in range(idx, horizon+idx):
            pred_f = self.net.forward(torch.tensor(data[-1], dtype=torch.get_default_dtype()
                                                   ).unsqueeze(0).to(self.net.device),
                                      dataset.__getitem__(i)[1].unsqueeze(0).to(
                                          self.net.device),
                                      dataset.__getitem__(i)[2].unsqueeze(0).to(
                                          self.net.device))[0].cpu().detach().numpy()
            pred_f = pred_f.reshape(2, int(data[0].shape[1]-2*dataset.off_set),
                                    int(data[0].shape[1]-2*dataset.off_set))

            prediction = data[-1][:, dataset.off_set:-dataset.off_set,
                                  dataset.off_set:-dataset.off_set] + dataset.delta_t*pred_f

            prediction = np.concatenate((left_bound[i+1, :, dataset.off_set:-dataset.off_set],
                                         prediction,
                                         right_bound[i+1, :, dataset.off_set:-dataset.off_set]), axis=2)
            prediction = np.concatenate((top_bound[i+1, :],
                                         prediction,
                                         bot_bound[i+1, :]), axis=1)
            prediction = svd.inverse_transform(
                svd.transform(prediction.reshape(1, -1)))
            data.append(prediction.reshape(2, int(np.sqrt(len(prediction[0])/2)),
                                           int(np.sqrt(len(prediction[0])/2))))
        return np.array(data)


def progress(train_loss, val_loss):
    return "Train/Loss: {:.8f} " \
           "Val/Loss: {:.8f}" \
           .format(train_loss, val_loss)
