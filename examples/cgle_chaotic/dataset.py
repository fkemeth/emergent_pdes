"""
Copyright © 2022 Felix P. Kemeth

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import diff as psdiff
from scipy.integrate import solve_ivp
from scipy.interpolate import interp2d

from lpde.dataset import Dataset
from lpde.utils import get_dudt_and_reshape_data

from typing import Any

import fun.dmaps as dmaps


def dudt(time: float,  # pylint: disable=unused-argument
         values: np.ndarray,
         length: float,
         c_1: float,
         c_2: float) -> np.ndarray:
    """
    Time derivative of the complex Ginzburg-Landau equation.

    Args:
        t: time step
        values: numpy array containing variables
        length: length of spatial domain
        c_1: parameter c1
        c_2: parameter c2

    Returns:
        Array with du/dt data
    """
    dydxx = psdiff(values.real, period=length, order=2) + \
        1.0j*psdiff(values.imag, period=length, order=2)
    return values - (1+1.0j*c_2)*np.abs(values)**2*values + (1+1.0j*c_1)*dydxx


def create_initial_conditions(n_grid_points: int) -> np.ndarray:
    """
    Specify initial conditions.

    Args:
        n_grid_points: number of spatial grid points

    Returns:
        Array with initial values
    """
    return 0.5 * np.random.randn(n_grid_points) + \
        0.5j * np.random.randn(n_grid_points)


def integrate(n_grid_points: int = 256,
              n_time_steps: int = 200,
              t_min: float = 1000.0,
              t_max: float = 1200.0,
              pars: Any = None):
    """
    Integrate complex Ginzburg-Landau equation.

    Args:
        n_grid_points: number of spatial grid points
        n_time_steps: number of time steps to sample data from
        t_min: start of time window
        t_max: end of time window
        pars: list of system parameters containing:
            length: length of spatial domain
            c_1: parameter c1
            c_2: parameter c2

    """
    # Default parameters if none are passed
    pars = [80, 0.0, -3.0] if pars is None else pars
    length, c_1, c_2 = pars

    # Write the parameters into a dictionary for future use.
    data_dict = {}
    data_dict['c_1'] = c_1
    data_dict['c_2'] = c_2
    data_dict['length'] = length
    data_dict['n_grid_points'] = n_grid_points
    data_dict['t_min'] = t_min
    data_dict['t_max'] = t_max
    data_dict['n_time_steps'] = n_time_steps

    # Set initial_conditions.
    initial_condition = create_initial_conditions(n_grid_points)
    data_dict['initial_condition'] = initial_condition

    # Set time vector.
    t_eval = np.linspace(t_min, t_max, n_time_steps+1, endpoint=True)

    # Compute solution.
    print('Computing the solution.')
    sol = solve_ivp(dudt,
                    [0, t_eval[-1]],
                    initial_condition,
                    t_eval=t_eval,
                    args=(length, c_1, c_2))
    data_dict['data'] = sol.y.T
    return data_dict


class CGLEDataset(Dataset):
    """
    CGLE dataset.

    Args:
        config: configfile with dataset parameters
    """

    def create_data(self) -> list:
        """
        Create partial differential equation dataset.

        Returns:
            Array with snapshot data
            Array with delta_x data
            Array with du/dt data
        """
        # Integrate pde
        pars = [self.config.getfloat('length'),
                self.config.getfloat('c_1'),
                self.config.getfloat('c_2')]
        data_dict = integrate(n_grid_points=self.config.getint('n_grid_points'),
                              n_time_steps=self.config.getint('n_time_steps'),
                              t_min=self.config.getfloat('tmin'),
                              t_max=self.config.getfloat('tmax'),
                              pars=pars)

        # If data type is complex, transform to real data
        if data_dict['data'].dtype == 'complex':
            data = np.stack(
                (data_dict['data'].real, data_dict['data'].imag), axis=-1)
        else:
            data = data_dict['data']

        delta_t = (self.config.getfloat('tmax')-self.config.getfloat('tmin')) / \
            self.config.getint('n_time_steps')

        delta_x = np.repeat(data_dict['length'] /
                            data_dict['n_grid_points'], len(data))

        x_data, delta_x, y_data = get_dudt_and_reshape_data(data,
                                                            delta_x,
                                                            delta_t,
                                                            self.config.getint('fd_dt_acc'))

        # Introduce variable dimension if it does not exist yet
        if len(x_data.shape) == 2:
            return x_data[:, np.newaxis], delta_x, y_data[:, np.newaxis]
        return np.transpose(x_data, (0, 2, 1)), delta_x, np.transpose(y_data, (0, 2, 1))

    def extract_emergent_space_coordinate(self) -> np.ndarray:
        _, evecs = dmaps.dmaps(np.transpose(
            self.x_data[:, 0]+1.0j*self.x_data[:, 1]), evals=20, eps=100, alpha=1)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(evecs[:, 1], evecs[:, 2])
        ax.set_xlabel('$v_1$')
        ax.set_ylabel('$v_2$')
        ax.set_title('diffusion maps embedding')
        plt.savefig('fig/diffusion_maps_embedding.pdf')
        plt.show()

        v_scaled = np.angle(evecs[:, 1]+1.0j*evecs[:, 2])

        v_scaled = np.roll(v_scaled, -np.argmin(v_scaled))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(v_scaled)
        ax.set_xlabel('$i$')
        ax.set_ylabel('$v$')
        ax.set_title('emergent coordinate')
        plt.savefig('fig/emergent_coordinate.pdf')
        plt.close('all')

        return v_scaled

    def emergent_space_embedding(self, v_scaled: np.ndarray) -> None:
        delta_t = (self.config.getfloat('tmax')-self.config.getfloat('tmin')) / \
            self.config.getint('n_time_steps')

        fd_dt_acc = self.config.getint('fd_dt_acc')

        tt_arr = np.linspace(self.config.getfloat('tmin'),
                             self.config.getfloat('tmax'),
                             self.config.getint('n_time_steps')+1
                             )[int(fd_dt_acc/2):-int(fd_dt_acc/2)]

        phi_arr = np.linspace(min(v_scaled), max(v_scaled), self.x_data.shape[-1], endpoint=True)

        fit_phi_r = interp2d(v_scaled, tt_arr,
                             self.x_data[:, 0, :], kind='cubic')
        fit_phi_i = interp2d(v_scaled, tt_arr,
                             self.x_data[:, 1, :], kind='cubic')

        x_data = np.stack((fit_phi_r(phi_arr, tt_arr),
                           fit_phi_i(phi_arr, tt_arr)),
                          axis=1)

        delta_x = np.repeat((np.max(v_scaled)-np.min(v_scaled))/len(phi_arr), len(x_data))

        self.x_data, self.delta_x, self.y_data = \
            get_dudt_and_reshape_data(x_data, delta_x, delta_t, fd_dt_acc)
