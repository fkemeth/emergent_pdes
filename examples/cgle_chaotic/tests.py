import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import findiff


def visualize_derivatives(network, dataset, path, idx=10):
    """Plot the calculated derivatives."""
    xx = dataset.config.getfloat("length") * \
        np.linspace(0, 1, num=dataset.config.getint("n_grid_points"), endpoint=False)

    input_x, dx, _ = dataset[idx]

    padding = network.get_off_set()
    if dataset.boundary_conditions == 'periodic':
        input_x = F.pad(input_x.unsqueeze(0), (padding, padding), mode='circular')[0]
    elif dataset.boundary_conditions == 'no-flux':
        input_x = F.pad(input_x.unsqueeze(0), (padding, padding), mode='reflect')[0]

    derivs = network.calc_derivs(input_x.unsqueeze(0).to(network.device),
                                 dx.unsqueeze(0).to(network.device)).detach().cpu().numpy()[0].T

    input_x, dx, _ = dataset[idx]

    padding = int(dataset.config.getint("n_grid_points")/2)

    if dataset.boundary_conditions == 'periodic':
        input_x = F.pad(input_x.unsqueeze(0), (padding, padding), mode='circular')[0]
    elif dataset.boundary_conditions == 'no-flux':
        input_x = F.pad(input_x.unsqueeze(0), (padding, padding), mode='reflect')[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(1, network.n_derivs+1):
        ax.plot(xx, derivs[:, i], label='Network derivative '+str(i))
        d_dx = findiff.FinDiff(0, float(dx.numpy()), i, acc=4)
        derivs_fd = d_dx(input_x.numpy()[0])[padding:-padding]
        ax.plot(xx, derivs_fd, '--', label='Numeric derivative '+str(i))
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'd/dx')
    plt.legend()
    plt.savefig(path + 'derivatives.pdf')
    plt.close()


def visualize_dynamics(dataset, path):
    xx = dataset.config.getfloat("length") * \
        np.linspace(0, 1, num=dataset.config.getint("n_grid_points"), endpoint=False)

    tt = np.linspace(0,
                     dataset.config.getint("n_time_steps")*dataset.delta_t,
                     dataset.config.getint("n_time_steps")+1)

    if dataset.use_fd_dt:
        tt = tt[:-dataset.config.getint("fd_dt_acc")]
        trajectory = dataset.x_data[
            :dataset.config.getint("n_time_steps")-dataset.config.getint("fd_dt_acc")+1, 0]
        dudt_trajectory = dataset.y_data[
            :dataset.config.getint("n_time_steps")-dataset.config.getint("fd_dt_acc")+1, 0]
    else:
        trajectory = dataset.x_data[:dataset.config.getint("n_time_steps")+1, 0]
        dudt_trajectory = dataset.y_data[:dataset.config.getint("n_time_steps")+1, 0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pl1 = ax.pcolor(xx, tt, trajectory, rasterized=True)
    ax.set_xlabel(r'x')
    ax.set_ylabel(r't')
    plt.colorbar(pl1, label='u')
    plt.savefig(path+'space_time_dynamics.pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pl1 = ax.pcolor(xx, tt, dudt_trajectory, rasterized=True)
    ax.set_xlabel(r'x')
    ax.set_ylabel(r't')
    plt.colorbar(pl1, label='du/dt')
    plt.savefig(path+'space_time_dudt.pdf')
    plt.close()


def visualize_learned_dudt(dataset, model, path, epoch=None, idx=10):
    xx = dataset.config.getfloat("length") * \
        np.linspace(0, 1, num=dataset.config.getint("n_grid_points"), endpoint=False)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xx, dataset.y_data[idx, 0], label='Finite difference du/dt')
    spatial_dimensions = dataset.x_data[idx].shape[1:]
    learned_dt = model.dfdt(0, dataset.x_data[idx], dataset.delta_x[idx], spatial_dimensions)
    ax.plot(xx, np.reshape(learned_dt, (model.net.n_vars, -1))[0], label='Learned du/dt')
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'd/dt')
    plt.legend()
    plt.savefig(path + 'learned_dudt.pdf')
    plt.close()


def visualize_predictions(dataset, model, path):
    t_eval = np.linspace(0,
                         dataset.config.getfloat('tmax')-dataset.config.getfloat('tmin'),
                         dataset.config.getint('n_time_steps')+1, endpoint=True)

    initial_condition, delta_x, _ = dataset[0]

    _, predictions = model.integrate(initial_condition.detach().numpy(),
                                     [delta_x.detach().numpy()],
                                     t_eval=t_eval)

    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(121)
    pl1 = ax1.pcolor(dataset.x_data[::5, 0], rasterized=True)
    ax1.set_xlabel(r'$x_i$')
    ax1.set_ylabel(r'$t_i$')
    plt.title('test data')
    plt.colorbar(pl1, label='$u$')
    ax2 = fig.add_subplot(122)
    pl2 = ax2.pcolor(predictions[::5, 0], rasterized=True)
    ax2.set_xlabel(r'$x_i$')
    ax2.set_ylabel(r'$t_i$')
    plt.title('prediction')
    plt.colorbar(pl2, label='$u$')
    plt.subplots_adjust(wspace=0.35)
    plt.savefig(path+'space_time_predictions.pdf')
    plt.show()
