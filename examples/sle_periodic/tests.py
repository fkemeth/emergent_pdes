import torch
import pickle
import findiff
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist


def test_perturbation(path='run1/dat/', idx=0):
    """Test if the integration of the perturbed dynamics works."""
    pkl_file = open(path+'/dat/run'+str(idx) + '_p_'+str(0)+'.pkl', 'rb')
    data_unperturbed = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open(path+'/dat/run'+str(idx) + '_p_'+str(-1)+'.pkl', 'rb')
    data_perturbed_neg = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open(path+'/dat/run'+str(idx) + '_p_'+str(1)+'.pkl', 'rb')
    data_perturbed_pos = pickle.load(pkl_file)
    pkl_file.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.linalg.norm(data_unperturbed["data"] -
                           data_perturbed_pos["data"], axis=1), label='d pos')
    ax.plot(np.linalg.norm(data_unperturbed["data"] -
                           data_perturbed_neg["data"], axis=1), label='d neg')
    ax.set_xlabel('t')
    ax.set_ylabel('d')
    plt.savefig(path + '/tests/dist_eucl.pdf')
    plt.show()

    print("Calculating closest distances....")
    dists_pos = cdist(np.append(data_perturbed_pos["data"].real, data_perturbed_pos["data"].imag,
                                axis=1), np.append(
        data_unperturbed["data"].real, data_unperturbed["data"].imag, axis=1))

    dists_neg = cdist(np.append(data_perturbed_neg["data"].real, data_perturbed_neg["data"].imag,
                                axis=1), np.append(
        data_unperturbed["data"].real, data_unperturbed["data"].imag, axis=1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.min(dists_pos, axis=1), label='d pos')
    ax.plot(np.min(dists_neg, axis=1), label='d neg')
    ax.set_xlabel('t')
    ax.set_ylabel('d')
    plt.savefig(path + '/tests/dist_closest.pdf')
    plt.show()


def test_dt(f_func, path='run1/dat/', idx=0):
    """Test if finite difference approximation of dt holds."""
    pkl_file = open(path+'/dat/run'+str(idx) + '_p_'+str(0)+'.pkl', 'rb')
    data_unperturbed = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open(path+'/dat/run'+str(idx) + '_p_'+str(-1)+'.pkl', 'rb')
    data_perturbed_neg = pickle.load(pkl_file)
    pkl_file.close()

    pars = data_unperturbed["pars"]
    if "omega" in pars.keys():
        pars["omega"] = data_unperturbed["omega"]

    delta_t = (data_unperturbed["tmax"]-data_unperturbed["tmin"])/data_unperturbed["T"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data_unperturbed["xx"], f_func(0, data_unperturbed["data"][0],
                                           pars).real, label="True")
    ax.plot(data_unperturbed["xx"],
            (data_unperturbed["data"][1].real-data_unperturbed["data"][0].real)/delta_t,
            label="Finite difference")
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel('dw/dt')
    plt.legend()
    plt.savefig(path + '/tests/dt_approx.pdf')
    plt.show()

    data_euler = np.zeros_like(data_perturbed_neg["data"])
    data_euler[0] = data_perturbed_neg["data"][0]


def test_dataset(dataset, path='run1/dat/'):
    """Test if dataset prepare works."""
    left_boundary_train, x_train, right_boundary_train, delta_x, y_train, _ = dataset.get_data(True)
    Nl, Nx, Nr = left_boundary_train.shape[2], x_train.shape[2], right_boundary_train.shape[2]

    print(delta_x.shape)

    assert y_train.shape[0] == x_train.shape[0]
    assert y_train.shape[2] == x_train.shape[2]
    assert y_train.shape[0] == delta_x.shape[0]

    print('mean dx: '+str(np.mean(delta_x)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(Nl), left_boundary_train[0, 0, :], color='red')
    ax.plot(np.arange(Nl, Nl+Nx), x_train[0, 0, :], color='green')
    ax.plot(np.arange(Nl+Nx, Nl+Nx+Nr), right_boundary_train[0, 0, :], color='red')
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel('Re W')
    plt.savefig(path + '/tests/dataset_real.pdf')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(Nl), left_boundary_train[0, 1, :], color='red')
    ax.plot(np.arange(Nl, Nl+Nx), x_train[0, 1, :], color='green')
    ax.plot(np.arange(Nl+Nx, Nl+Nx+Nr), right_boundary_train[0, 1, :], color='red')
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel('Re W')
    plt.savefig(path + '/tests/dataset_imag.pdf')
    plt.show()


def test_svd(dataset_train, dataset_test, path='run1/dat/'):
    """Test if dataset prepare works."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dataset_train.x_data[0, 0], label='original')
    ax.plot(dataset_train.svd.inverse_transform(
        dataset_train.svd.transform(dataset_train.x_data[0].reshape(1, -1))).reshape(2, -1)[0], '--',
        label='svd')
    ax.plot(dataset_test.x_data[0, 0], label='original')
    ax.plot(dataset_train.svd.inverse_transform(
        dataset_train.svd.transform(dataset_test.x_data[0].reshape(1, -1))).reshape(2, -1)[0], '--',
        label='svd')
    ax.set_xlabel(r'i')
    ax.set_ylabel('Re W')
    plt.savefig(path + '/tests/svd.pdf')
    plt.show()


def test_fd_coeffs(network, path='run1/dat/'):
    """Plot finite difference coefficients."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pl1 = ax.pcolor(network.coeffs.cpu().numpy()[:, 0, :], rasterized=True)
    ax.set_xlabel(r'i')
    ax.set_ylabel(r'j')
    plt.colorbar(pl1)
    plt.savefig(path + '/tests/fd_coeffs.pdf')
    plt.show()


def test_derivs(network, input_x, dx, path='run1/dat/'):
    """Plot finite difference coefficients."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    derivs = network.calc_derivs(input_x.to(network.device),
                                 dx.to(network.device)).cpu().numpy()[0].T
    for i in range(1, network.n_derivs+1):
        ax.plot(np.arange(int((network.kernel_size-1)/2),
                          input_x.shape[-1]-int((network.kernel_size-1)/2)),
                derivs[:, i], color=colors[i])
        d_dx = findiff.FinDiff(0, dx.numpy()[0], i, acc=4)
        ax.plot(np.arange(input_x.shape[-1]), d_dx(input_x[0, 0].numpy()), '--', color=colors[i])
    ax.set_xlabel(r'i')
    ax.set_ylabel(r'd/dx')
    plt.savefig(path + '/tests/derivatives.pdf')
    plt.show()


def test_learned_dt(model, dataset, f_func, path='run1/dat/', idx=0):
    """Test if finite difference approximation of dt holds."""
    pkl_file = open(path+'/dat/run'+str(0) + '_p_'+str(-1)+'.pkl', 'rb')
    data_dict = pickle.load(pkl_file)
    pkl_file.close()

    delta_t = (data_dict["tmax"]-data_dict["tmin"])/data_dict["T"]

    pars = data_dict["pars"]
    if "omega" in pars.keys():
        gamma = dataset.param[idx]*0.02+1.75
        pars["omega"] = np.linspace(-gamma, gamma, len(dataset.x_data[idx][0])) + 0.2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(data_dict["xx"], f_func(0, dataset.x_data[idx][0]+1.0j*dataset.x_data[idx][1],
    #                                 pars).real, label="True")
    ax.plot(data_dict["xx"],
            (dataset.x_data[idx+1][0]-dataset.x_data[idx][0])/delta_t,
            label="Finite difference")
    # ax.plot(data_dict["xx"][dataset.off_set:-dataset.off_set],
    #         dataset.__getitem__(idx)[2][0].numpy(), label="Finite difference loader")
    ax.plot(data_dict["xx"][dataset.off_set:-dataset.off_set],
            model.net.forward(dataset.__getitem__(idx)[0].unsqueeze(0).to(model.net.device),
                              dataset.__getitem__(idx)[1].unsqueeze(0).to(model.net.device),
                              dataset.__getitem__(idx)[3].to(model.net.device)
                              )[0, 0].cpu().detach().numpy(),
            label="Learned")
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel('dw/dt')
    plt.legend()
    plt.savefig(path + '/tests/learned_dt.pdf')
    plt.show()


def test_integration(model, dataset, svd, idx, horizon, path='run1/dat/'):
    """Inegrate snapshot using the learned model."""
    prediction = model.integrate(dataset, svd, idx, horizon)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.pcolor(prediction[:, 0, :], vmin=-.1, vmax=.1, rasterized=True)
    ax.set_xlabel('i')
    ax.set_ylabel('t')
    plt.title('learned')
    ax2 = fig.add_subplot(122)
    ax2.pcolor(dataset.x_data[idx:idx+horizon, 0, :], vmin=-.1, vmax=.1, rasterized=True)
    ax2.set_xlabel('i')
    ax2.set_ylabel('')
    plt.title('true')
    plt.savefig(path + '/tests/integration_model.pdf')
    plt.show()
    return prediction


def test_transient_dynamics(model, dataset, svd, idx, t_off, path='run1/dat/'):
    """Inegrate snapshot using the learned model."""
    pkl_file = open(path+'/dat/run'+str(idx) + '_p_'+str(0)+'.pkl', 'rb')
    data_unperturbed = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open(path+'/dat/run'+str(idx) + '_p_'+str(-1)+'.pkl', 'rb')
    data_perturbed_neg = pickle.load(pkl_file)
    pkl_file.close()

    prediction = model.integrate(dataset, svd, t_off, data_unperturbed["T"]-t_off-1)

    print("Calculating closest distances....")
    dists_neg = cdist(np.append(data_perturbed_neg["data"].real, data_perturbed_neg["data"].imag,
                                axis=1), np.append(
        data_unperturbed["data"].real, data_unperturbed["data"].imag, axis=1))

    dists_learned = cdist(np.append(prediction[:, 0], prediction[:, 1], axis=1), np.append(
        data_unperturbed["data"].real, data_unperturbed["data"].imag, axis=1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.min(dists_neg, axis=1)[t_off:], label='d true')
    ax.plot(np.min(dists_learned, axis=1), label='d learned')
    plt.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('d')
    plt.savefig(path + '/tests/dist_closest_learned.pdf')
    plt.show()
