import torch

import numpy as np

from int.cgle_1d import integrate


class Dataset(torch.utils.data.Dataset):
    """Phase field front dataset."""

    def __init__(self, config, n_runs, start_idx=0, downsample=False):
        self.boundary_conditions = config["boundary_conditions"]
        self.initial_conditions = config["ic"]
        self.use_fd_dt = config.getboolean("use_fd_dt")
        if not self.use_fd_dt:
            self.dudt = config["dudt"]
        self.n_runs = n_runs
        self.delta_t = float(config["dt"])
        self.rescale_dx = float(config["rescale_dx"])
        self.config = config
        self.start_idx = start_idx
        self.x_data, self.delta_x, self.y_data = self.create_data()
        if downsample:
            self.x_data = self.x_data[::downsample]
            self.delta_x = self.delta_x[::downsample]
            self.y_data = self.y_data[::downsample]
        self.n_samples = self.x_data.shape[0]

    def create_data(self):
        x_data = []
        delta_x = []
        y_data = []
        for i in range(self.start_idx, self.start_idx+self.n_runs):
            if self.config.getboolean("load_data"):
                data = np.load('data/run_'+str(i)+'.npy')
                dx = np.load('data/dx_'+str(i)+'.npy')
            else:
                data_dict = integrate(dynamics='intermittency',
                                      L=float(self.config["L"]), N=int(self.config["N"]),
                                      tmin=float(self.config["tmin"]),
                                      tmax=float(self.config["tmax"]),
                                      dt=float(self.config["dt"]),
                                      ic=self.initial_conditions)
                # Perturb Snapshot
                u0 = data_dict["data"][-1]
                x = float(self.config["L"])*np.linspace(0, 1, num=int(self.config["N"]),
                                                        endpoint=False)
                for k in range(20):
                    u0 += float(self.config["perturbation"])*1.0j*np.random.randn()*np.sin(
                        np.random.randint(int(k*2),
                                          int((k+1)*2))*x*2*np.pi/float(self.config["L"])) + \
                        float(self.config["perturbation"])*1.0j*np.random.randn()*np.sin(
                            np.random.randint(int(k*2), int((k+1)*2))*x*2*np.pi/float(self.config["L"]))
                u0 += float(self.config["perturbation"])*np.random.randn(u0.shape[0]) + \
                    float(self.config["perturbation"])*np.random.randn(u0.shape[0])

                data_dict = integrate(dynamics='intermittency',
                                      L=float(self.config["L"]), N=int(self.config["N"]),
                                      tmin=0,
                                      tmax=float(self.config["tmax"])-float(self.config["tmin"]),
                                      dt=float(self.config["dt"]),
                                      ic='manual',
                                      Ainit=u0)
                if data_dict["data"].dtype == 'complex':
                    data = np.stack((data_dict["data"].real, data_dict["data"].imag), axis=-1)
                else:
                    data = data_dict["data"]
                dx = data_dict["L"]/data_dict["N"]
                np.save('data/run_'+str(i)+'.npy', data)
                np.save('data/dx_'+str(i)+'.npy', dx)

            if self.use_fd_dt:
                if int(self.config["fd_dt_acc"]) == 2:
                    # accuracy 2
                    y_data.append((data[2:]-data[:-2])/(2*self.delta_t))
                    x_data.append(data[1:-1])
                    delta_x.append(np.repeat(dx, len(data)-2))
                elif int(self.config["fd_dt_acc"]) == 4:
                    # accuracy 4
                    y_data.append((data[:-4]-8*data[1:-3]+8 *
                                   data[3:-1]-data[4:])/(12*self.delta_t))
                    x_data.append(data[2:-2])
                    delta_x.append(np.repeat(dx, len(data)-4))
                else:
                    raise ValueError("Finite difference in time accuracy must be 2 or 4.")
            else:
                raise ValueError("Using true du/dt not implemented!")
                # y_data.append([self.config["dudt"](0, u, pars) for u in data])
                # x_data.append(data)
                # delta_x.append(np.repeat(dx, len(data)))

        x_data = np.concatenate(x_data, axis=0)
        y_data = np.concatenate(y_data, axis=0)
        delta_x = np.concatenate(delta_x, axis=0)*self.rescale_dx
        if len(x_data.shape) == 2:
            return x_data[:, np.newaxis], delta_x, y_data[:, np.newaxis]
        else:
            return np.transpose(x_data, (0, 2, 1)), delta_x, np.transpose(y_data, (0, 2, 1))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        _x = torch.tensor(self.x_data[index], dtype=torch.get_default_dtype())
        _dx = torch.tensor(self.delta_x[index], dtype=torch.get_default_dtype())
        _y = torch.tensor(self.y_data[index], dtype=torch.get_default_dtype())
        return _x, _dx, _y
