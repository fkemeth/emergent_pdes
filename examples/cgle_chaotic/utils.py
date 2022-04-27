from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate

import fun.dmaps as dmaps


def dmaps_transform(n_total, dataset: Dataset):
    """Parametrize data using first diffusion mode."""
    _, evecs = dmaps.dmaps(np.transpose(
        dataset.x_data[:, 0]+1.0j*dataset.x_data[:, 1]), eps=100, alpha=1)
    np.save('data/evecs.npy', evecs)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(evecs[:, 1], evecs[:, 2])
    ax.set_xlabel(r'$\phi_1$')
    ax.set_ylabel(r'$\phi_2$')
    plt.savefig('fig/dmaps_emb.pdf')
    plt.close()

    v_scaled = np.angle(evecs[:, 1]+1.0j*evecs[:, 2])
    v_scaled = np.roll(v_scaled, -np.argmin(v_scaled))
    for data_idx in range(n_total):
        print(data_idx)
        data = np.load('data/run_'+str(data_idx)+'.npy')
        np.save('data/run_'+str(data_idx)+'_orig.npy', data)
        tt_arr = np.arange(data.shape[0])
        phi_arr = np.linspace(np.min(v_scaled), np.max(v_scaled), data.shape[1], endpoint=False)
        fit_phi_r = interpolate.interp2d(v_scaled, tt_arr,
                                         data[:, :, 0], kind='cubic')
        fit_phi_i = interpolate.interp2d(v_scaled, tt_arr,
                                         data[:, :, 1], kind='cubic')
        data = np.stack((fit_phi_r(phi_arr, tt_arr), fit_phi_i(phi_arr, tt_arr)), axis=-1)
        np.save('data/run_'+str(data_idx)+'.npy', data)
        np.save('data/dx_'+str(data_idx)+'.npy', np.array(phi_arr[1]-phi_arr[0]))

    print('Saveing v_scaled.')
    np.save('data/v_scaled.npy', v_scaled)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(v_scaled)
    ax.set_xlabel('$i$')
    ax.set_ylabel(r'$\tilde{x}$')
    plt.savefig('fig/v_scaled.pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolor(data[::4, :, 0], rasterized=True)
    ax.set_xlabel('$i$')
    ax.set_ylabel('$t$')
    plt.savefig('fig/data_transformed.pdf')
    plt.close()
