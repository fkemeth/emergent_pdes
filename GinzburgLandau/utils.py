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

# import cudamat as cm

from sklearn.decomposition import TruncatedSVD
from scipy.integrate import ode
from scipy import interpolate

import int.matthews as mint
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


def perturb_limit_cycle(Ad, n, config):
    """Get the slow directions of the Matthews system."""
    n_per = get_period(np.linalg.norm(Ad["data"][int(n*int(config["T_off"])):] -
                                      Ad["data"][int(n*int(config["T_off"])), :], axis=1))

    print("Period is "+str(n_per))
    dt_per = (Ad["tmax"]-Ad["tmin"])/Ad["T"]

    print("Calculating monodromy matrix.")
    t_start = n*int(config["T_off"])
    progress_bar = tqdm.tqdm(range(t_start, t_start+n_per),
                             total=n_per, leave=True)

    cm.cublas_init()
    v_mono = cm.CUDAMatrix(np.eye(2*int(config["N_int"])))

    # for i in range(t_start, t_start+n_per):
    for i in progress_bar:
        # print(i)
        # jacobian = cm.CUDAMatrix(mint.jac(0, Ad["data"][i], Ad["pars"]))
        # v_mono = v_mono + dt_per*cm.dot(jacobian, v_mono)
        jacobian = cm.CUDAMatrix(mint.jac(0, Ad["data"][i], Ad["pars"]))
        v_mono.add(cm.dot(jacobian, v_mono).mult(dt_per), target=v_mono)

    evals, evecs = np.linalg.eig(v_mono.asarray())
    # v_mono = np.eye(2*int(config["N_int"]))
    # t_start = n*int(config["T_off"])
    # for i in range(t_start, t_start+n_per):
    #     # print(i)
    #     jacobian = mint.jac(0, Ad["data"][i], Ad["pars"])
    #     v_mono = v_mono + dt_per*np.dot(jacobian, v_mono)

    # evals, evecs = np.linalg.eig(v_mono)
    idx = np.abs(evals).argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    assert (np.abs(evals[0]-1.0) <
            1e-1), "Neutral direction, D[0]="+str(evals[0])
    v_tmp = evecs[:, 1]+evecs[:, 2]
    v_tmp = v_tmp/np.linalg.norm(v_tmp)
    return v_tmp[:Ad["N"]]+1.0j*v_tmp[Ad["N"]:]


def perturb_fixed_point(Ad, n, config):
    """Get the slow directions of the Matthews system."""
    jacobian = mint.jac(0, Ad["data"][n*int(config["T_off"])], Ad["pars"])
    evals, evecs = np.linalg.eig(jacobian)
    idx = np.real(evals).argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    assert np.real(
        evals[0]-evals[1]) < 1e-6, "First two eigenvalues should be complex conjugate."
    v_tmp = evecs[:, 0]+evecs[:, 1]
    v_tmp = v_tmp/np.linalg.norm(v_tmp)
    return v_tmp[:Ad["N"]]+1.0j*v_tmp[Ad["N"]:]


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


class Network(nn.Module):
    """Pytorch network architecture."""

    def __init__(self, config):
        super(Network, self).__init__()
        self.kernel_size = int(config["kernel_size"])
        self.n_derivs = int(config["n_derivs"])
        self.device = config['device']
        self.use_param = config.getboolean("use_param")
        self.hypervisc = config.getboolean("hypervisc")
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")
        self.register_buffer('coeffs', self.get_coeffs(max_deriv=self.n_derivs))
        if self.hypervisc:
            self.register_buffer('cvisc', self.get_coeffs(min_deriv=4, max_deriv=7))

        layers = []
        if self.use_param:
            num_features = int(2*(self.n_derivs+1)+1)
        else:
            num_features = int(2*(self.n_derivs+1))
        n_channels = int(config["n_filters"])
        # layers.append(nn.BatchNorm1d(num_features))
        for _ in range(int(config["n_layers"])):
            layers.append(nn.Conv1d(num_features, n_channels, [1], stride=1, padding=0, bias=True))
            # layers.append(nn.ReLU())
            layers.append(nn.Tanh())
            num_features = n_channels

        layers.append(nn.Conv1d(num_features, 2, [1], stride=1, padding=0, bias=True))

        self.network = nn.Sequential(*layers)
        self.trainable_parameters = sum(p.numel()
                                        for p in self.network.parameters() if p.requires_grad)

    # def get_coeffs(self):
    #     """Get finite difference coefficients."""
    #     coeffs = np.zeros((self.n_derivs+1, 1, self.kernel_size))
    #     # Zero'th derivative
    #     coeffs[0, 0, int((self.kernel_size-1)/2)] = 1.0
    #     # Higher derivatives
    #     for i in range(1, self.n_derivs+1):
    #         # Get coefficient for certain derivative with maximal acc order for given kernel_size
    #         fd_coeff = np.array([1.])
    #         acc_order = 0
    #         while len(fd_coeff) < self.kernel_size:
    #             acc_order += 2
    #             fd_coeff = findiff.coefficients(i, acc_order)["center"]["coefficients"]
    #         assert len(fd_coeff) == self.kernel_size, \
    #             "Finite difference coefficients do not match kernel"
    #         coeffs[i, 0, :] = fd_coeff
    #     return torch.tensor(coeffs, requires_grad=False).to(self.device)

    def get_coeffs(self, min_deriv=0, max_deriv=5):
        """Get finite difference coefficients."""
        assert max_deriv > min_deriv, "Derivative range not specified"
        assert min_deriv >= 0, "Derivatives should be larger zero"

        coeffs = np.zeros((max_deriv-min_deriv+1, 1, self.kernel_size))
        # Finite difference coefficients
        for i in range(min_deriv, max_deriv+1):
            # Get coefficient for certain derivative with maximal acc order for given kernel_size
            fd_coeff = np.array([1.])
            acc_order = 0
            while len(fd_coeff) < self.kernel_size:
                acc_order += 2
                fd_coeff = findiff.coefficients(i, acc_order)["center"]["coefficients"]
            assert len(fd_coeff) == self.kernel_size, \
                "Finite difference coefficients do not match kernel"
            coeffs[i, 0, :] = fd_coeff
        return torch.tensor(coeffs, requires_grad=False,
                            dtype=torch.get_default_dtype()).to(self.device)

    def calc_derivs(self, x, dx):
        """Calculate derivativers of input x."""
        # print(self.coeffs.dtype)
        # print(x.dtype)
        finite_diffs = torch.cat([F.conv1d(x[:, :1], self.coeffs),
                                  F.conv1d(x[:, 1:], self.coeffs)], dim=1).to(self.device)
        scales = torch.cat([torch.pow(dx.unsqueeze(1), i)
                            for i in range(self.coeffs.shape[0])], axis=-1)
        scales = torch.cat((scales, scales), axis=1)
        scales = scales.unsqueeze(2).repeat(1, 1, finite_diffs.shape[-1]).to(self.device)
        return finite_diffs/scales

    def forward(self, x, dx, param=None):
        """Predict temporal derivative of input x."""
        # Calculate derivatives
        x = self.calc_derivs(x, dx)
        # TODO normalize derivatives
        if self.hypervisc:
            x = x - x
        if self.use_param:
            x = torch.cat([x, param.unsqueeze(-1).unsqueeze(-1).repeat(1, 1,
                                                                       x.shape[-1])], axis=1)
        # Forward through distributed parameter stack
        x = self.network(x)
        return x


def progress(train_loss, val_loss):
    return "Train/Loss: {:.8f} " \
           "Val/Loss: {:.8f}" \
           .format(train_loss, val_loss)


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
        for (data, delta_x, target, param) in self.dataloader_train:
            data = data.to(self.device)
            delta_x = delta_x.to(self.device)
            target = target.to(self.device)
            if self.net.use_param:
                param = param.to(self.device)

            # backward
            self.optimizer.zero_grad()

            # forward
            if self.net.use_param:
                output = self.net(data, delta_x, param)
            else:
                output = self.net(data, delta_x)

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
            for (data, delta_x, target, param) in self.dataloader_val:
                data = data.to(self.device)
                delta_x = delta_x.to(self.device)
                target = target.to(self.device)
                if self.net.use_param:
                    param = param.to(self.device)

                # forward
                if self.net.use_param:
                    output = self.net(data, delta_x, param)
                else:
                    output = self.net(data, delta_x)

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

    def integrate(self, dataset, svd, idx, horizon, use_svd=True):
        """Integrate idx'th snapshot of dataset for horizon time steps."""
        # assert idx == 0, "Only idx = 0 implemented."
        left_bounds, _, right_bounds, _, _, param = dataset.get_data(True)
        data = []
        if use_svd:
            data0 = svd.inverse_transform(
                svd.transform(dataset.x_data[idx].reshape(1, -1)))
            data.append(data0.reshape(2, -1))
        else:
            data.append(dataset.x_data[idx])
        # data = []
        # data.append(dataset.x_data[idx])

        # assert np.all(data[0][:, :dataset.off_set] == left_bounds[idx])
        # assert np.all(data[0][:, -dataset.off_set:] == right_bounds[idx])

        for i in range(idx, horizon+idx):
            pred_f = self.net.forward(torch.tensor(data[-1], dtype=torch.get_default_dtype()
                                                   ).unsqueeze(0).to(self.net.device),
                                      dataset.__getitem__(i)[1].unsqueeze(0).to(
                self.net.device),
                torch.tensor(param[idx],
                             dtype=torch.get_default_dtype()).unsqueeze(0).to(self.net.device))[0].cpu().detach().numpy()
            prediction = data[-1][:, dataset.off_set:-dataset.off_set] + dataset.delta_t*pred_f

            prediction = np.concatenate((left_bounds[i+1], prediction, right_bounds[i+1]), axis=1)
            if use_svd:
                prediction = svd.inverse_transform(
                    svd.transform(prediction.reshape(1, -1)))
                data.append(prediction.reshape(2, -1))
            else:
                data.append(prediction)
        return np.array(data)

    def integrate_implicit(self, dataset, svd, idx, horizon):
        """Integrate idx'th snapshot of dataset for horizon time steps."""
        # assert idx == 0, "Only idx = 0 implemented."
        left_bounds, _, right_bounds, _, _ = dataset.get_data(True)
        data0 = svd.inverse_transform(
            svd.transform(dataset.x_data[idx].reshape(1, -1)))
        data = []
        data.append(data0.reshape(2, -1))

        # assert np.all(data[0][:, :dataset.off_set] == left_bounds[idx])
        # assert np.all(data[0][:, -dataset.off_set:] == right_bounds[idx])

        def f(t, y):
            y = np.vstack((y.real, y.imag))
            in_y = torch.tensor(y).unsqueeze(0).to(self.net.device)
            out_y = self.net.forward(in_y, dataset.__getitem__(i)[1].unsqueeze(0).to(
                self.net.device))[0].cpu().detach().numpy()
            return out_y[0, 0]+1.0j*out_y[0, 1]

        y0, t0 = data[-1][0]+1.0j*data[-1][1], 0
        print(y0.shape)
        r = ode(f).set_integrator('zvode', method='adams')
        r.set_initial_value(y0, t0)
        i = idx
        while i < horizon+idx:
            prediction = r.integrate(dataset.delta_t)
            # pred_f = self.net.forward(torch.tensor(data[-1]).unsqueeze(0).to(self.net.device),
            #                           dataset.__getitem__(i)[1].unsqueeze(0).to(
            #                               self.net.device))[0].cpu().detach().numpy()

            # prediction = data[-1][:, dataset.off_set:-dataset.off_set] + dataset.delta_t*pred_f
            prediction = np.vstack((prediction.real, prediction.imag))
            print(prediction.shape)
            prediction = np.concatenate((left_bounds[i+1], prediction, right_bounds[i+1]), axis=1)
            prediction = svd.inverse_transform(
                svd.transform(prediction.reshape(1, -1)))
            data.append(prediction.reshape(2, -1))
            i += 1
            r = ode(f).set_integrator('vode', method='adams')
            r.set_initial_value(data[-1][0]+1.0j*data[-1][1], 0)
        return np.array(data)

    def integrate_rk4(self, dataset, svd, idx, horizon):
        """Integrate idx'th snapshot of dataset for horizon time steps."""
        # assert idx == 0, "Only idx = 0 implemented."
        left_bounds, x_data, right_bounds, _, _, param = dataset.get_data(True)
        data0 = svd.inverse_transform(
            svd.transform(dataset.x_data[idx].reshape(1, -1)))
        data = []
        data.append(data0.reshape(2, -1))
        # data = []
        # data.append(dataset.x_data[idx])

        # assert np.all(data[0][:, :dataset.off_set] == left_bounds[idx])
        # assert np.all(data[0][:, -dataset.off_set:] == right_bounds[idx])

        for i in range(idx, horizon+idx):
            k1 = self.net.forward(torch.tensor(data[-1],
                                               dtype=torch.get_default_dtype()
                                               ).unsqueeze(0).to(self.net.device),
                                  dataset.__getitem__(i)[1].unsqueeze(0).to(self.net.device),
                                  torch.tensor(param[idx],
                                               dtype=torch.get_default_dtype()
                                               ).unsqueeze(0).to(self.net.device)
                                  )[0].cpu().detach().numpy()
            k2 = self.net.forward(torch.tensor(data[-1][:, dataset.off_set:-dataset.off_set] +
                                               dataset.delta_t*k1/2,
                                               dtype=torch.get_default_dtype()
                                               ).unsqueeze(0).to(self.net.device),
                                  dataset.__getitem__(i)[1].unsqueeze(0).to(self.net.device),
                                  torch.tensor(param[idx],
                                               dtype=torch.get_default_dtype()
                                               ).unsqueeze(0).to(self.net.device)
                                  )[0].cpu().detach().numpy()
            k3 = self.net.forward(torch.tensor(
                data[-1][:, int(2*dataset.off_set):-int(2*dataset.off_set)] +
                dataset.delta_t*k2/2,
                dtype=torch.get_default_dtype()).unsqueeze(0).to(self.net.device),
                dataset.__getitem__(i)[1].unsqueeze(0).to(
                    self.net.device),
                torch.tensor(param[idx], dtype=torch.get_default_dtype()).unsqueeze(0).to(
                self.net.device))[0].cpu().detach().numpy()
            k4 = self.net.forward(torch.tensor(
                data[-1][:, int(3*dataset.off_set):-int(3*dataset.off_set)] +
                dataset.delta_t*k3, dtype=torch.get_default_dtype()).unsqueeze(0).to(self.net.device),
                dataset.__getitem__(i)[1].unsqueeze(0).to(
                    self.net.device),
                torch.tensor(param[idx], dtype=torch.get_default_dtype()).unsqueeze(0).to(
                self.net.device))[0].cpu().detach().numpy()
            prediction = data[-1][:, int(4*dataset.off_set):-int(4*dataset.off_set)] + \
                dataset.delta_t/6.*(k1[:, int(3*dataset.off_set):-int(3*dataset.off_set)] +
                                    2*k2[:, int(2*dataset.off_set):-int(2*dataset.off_set)] +
                                    2*k3[:, int(dataset.off_set):-int(dataset.off_set)] +
                                    k4)

            prediction = np.concatenate((left_bounds[i+1], x_data[i+1, :, :int(3*dataset.off_set)],
                                         prediction, x_data[i+1, :, -int(3*dataset.off_set):],
                                         right_bounds[i+1]), axis=1)
            prediction = svd.inverse_transform(
                svd.transform(prediction.reshape(1, -1)))
            data.append(prediction.reshape(2, -1))
        return np.array(data)


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


class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
