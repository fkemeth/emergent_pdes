import torch
import pickle
import findiff
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist


def test_perturbation(path='data/', idx=0):
    """Test if the integration of the perturbed dynamics works."""
    pkl_file = open(path+str(idx) + '_p_'+str(0)+'.pkl', 'rb')
    data_unperturbed = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open(path+str(idx) + '_p_'+str(1)+'.pkl', 'rb')
    data_perturbed_neg = pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open(path+str(idx) + '_p_'+str(-1)+'.pkl', 'rb')
    data_perturbed_pos = pickle.load(pkl_file)
    pkl_file.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pl1 = ax.pcolor(data_perturbed_neg["data"][::10, ::8, 0]-data_unperturbed["data"][::10, ::8, 0])
    ax.set_xlabel('')
    ax.set_ylabel('')
    cbar = plt.colorbar(pl1)
    # plt.savefig('')
    plt.show()

    T = len(data_unperturbed["tt"])

    print("Calculating closest distances....")
    dists_pos = cdist(np.append(data_perturbed_pos["data"][:, :, 0],
                                data_perturbed_pos["data"][:, :, 1],
                                axis=1), np.append(
                                    data_unperturbed["data"][:, :, 0],
                                    data_unperturbed["data"][:, :, 1], axis=1))
    dists_neg = cdist(np.append(data_perturbed_neg["data"][:, :, 0],
                                data_perturbed_neg["data"][:, :, 1],
                                axis=1), np.append(
                                    data_unperturbed["data"][:, :, 0],
                                    data_unperturbed["data"][:, :, 1], axis=1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data_unperturbed["tt"], np.linalg.norm(data_unperturbed["data"] -
                                                   data_perturbed_pos["data"], axis=(1, 2)), label='d pos')
    ax.plot(data_unperturbed["tt"], np.linalg.norm(data_unperturbed["data"] -
                                                   data_perturbed_neg["data"], axis=(1, 2)), label='d neg')
    # ax.plot(data_unperturbed["tt"], np.linalg.norm(data_unperturbed["data_fitted"] -
    #                                                data_perturbed_pos["data_fitted"], axis=(1, 2)), label='d pos')
    # ax.plot(data_unperturbed["tt"], np.linalg.norm(data_unperturbed["data_fitted"] -
    #                                                data_perturbed_neg["data_fitted"], axis=(1, 2)), label='d neg')
    ax.set_xlabel('t')
    ax.set_ylabel('d')
    plt.savefig(path + '/tests_dist_eucl.pdf')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data_unperturbed["tt"], np.min(dists_pos, axis=1), label='d pos')
    ax.plot(data_unperturbed["tt"], np.min(dists_neg, axis=1), label='d neg')
    # ax.plot(np.linalg.norm(data_unperturbed["data"] -
    #                        data_perturbed_neg["data"], axis=(1, 2)), label='d neg')
    # ax.plot(np.linalg.norm(data_unperturbed["data_fitted"] -
    #                        data_perturbed_pos["data_fitted"], axis=(1, 2)), label='d pos')
    # ax.plot(np.linalg.norm(data_unperturbed["data_fitted"] -
    #                        data_perturbed_neg["data_fitted"], axis=(1, 2)), label='d neg')
    ax.set_xlabel('t')
    ax.set_ylabel('d')
    plt.savefig(path + '/tests_dist.pdf')
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
