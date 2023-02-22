import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from scipy import interpolate


from utils import config
from fitting import BasisFit, BasisFitter


def R(fineState, fitter):
    fineState = fineState.T
    # fineState = fineState.reshape(2, 1024)
    coeffs = [fitter.fit(variable).coeffs for variable in fineState]
    return np.hstack(coeffs)


def phase_space_vid():
    i = 0
    p = config["DATA"]["p_list"][0]
    pkl_file = open(config["DATA"]["path"]+str(i) + '_p_'+str(p)+'.pkl', 'rb')
    pkl_data = pickle.load(pkl_file)
    pkl_file.close()

    ids = np.arange(len(pkl_data["xx"]))
    np.random.shuffle(ids)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'$V_i$')
    ax1.set_ylabel(r'$h_i$')
    scas = []
    for i in range(0, int(len(pkl_data["tt"])), 50):
        sca1 = ax1.scatter(pkl_data["data"][i, ids, 1],
                           pkl_data["data"][i, ids, 0], zorder=10, color='black', s=5)
        scas.append([sca1])

    ani = animation.ArtistAnimation(
        fig, scas, interval=100, blit=True, repeat_delay=0, repeat=True)
    writergif = animation.PillowWriter(fps=20)
    ani.save(config["DATA"]["path"] + 'phase_space_vid.gif', writer=writergif)
    plt.show()


def temporal_evolution_vid():
    i = 0
    p = config["DATA"]["p_list"][0]
    pkl_file = open(config["DATA"]["path"]+str(i) + '_p_'+str(p)+'.pkl', 'rb')
    pkl_data = pickle.load(pkl_file)
    pkl_file.close()

    ids = np.arange(len(pkl_data["xx"]))
    np.random.shuffle(ids)

    D = np.load(config["DATA"]["path"]+'dmaps_D.npy')
    V = np.load(config["DATA"]["path"]+'dmaps_V.npy')

    v_scaled1 = V[:, 1]
    v_scaled2 = V[:, 2]

    # pkl_data["data"] = pkl_data["data"][:, np.argsort(v_scaled1), :]
    # pkl_data["data"] = pkl_data["data"][:, :, np.argsort(v_scaled2)]
    # v_scaled1 = np.sort(v_scaled1)
    # v_scaled2 = np.sort(v_scaled2)

    Xfull, Yfull = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(0, 1, 64))

    maxOrd = 2
    hets = np.vstack((v_scaled1, v_scaled2)).T
    fitter = BasisFitter(
        hets, maxOrd=maxOrd,
        basis=None
    )
    coeffs = R(pkl_data["data"][0, :, :], fitter)
    # coeffs = coeffs.reshape(int(coeffs.size/2), 2).T
    coeffs = coeffs.reshape(2,  int(coeffs.size/2))
    fit = BasisFit(coeffs[0, :], fitter.basis)
    Zfull = fit(Xfull, Yfull)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xfull, Yfull, Zfull)
    ax.plot_surface(Xfull, Yfull, Zfull, alpha=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # plt.savefig('')
    plt.show()

    # fit_phi_r = interpolate.Rbf(v_scaled1, v_scaled2, pkl_data["data"][0, :, 0], smooth=100)
    # fit_phi_r = interpolate.SmoothBivariateSpline(v_scaled1, v_scaled2, pkl_data["data"][0, :, 0])
    # fit_phi_r = interpolate.SmoothBivariateSpline(v_scaled1[v_scaled2 > 0], v_scaled2[v_scaled2 > 0],
    #                                               pkl_data["data"][0, v_scaled2 > 0, 0])

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(Xfull, Yfull, Zfull)
    # ax.plot_surface(Xfull, Yfull, fit_phi_r(Xfull.flatten(),
    #                                         Yfull.flatten(), grid=False).reshape(64, 64), alpha=0.5)
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    # # plt.savefig('')
    # plt.show()

    # plt.pcolor(v_scaled1[::4], v_scaled2[::4], Zfull[::4, ::4])
    # plt.colorbar()
    # plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel(r'$i$')
    ax1.set_ylabel(r'$h_i$')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlabel(r'$\phi_1$')
    ax2.set_ylabel(r'$\phi_2$')
    ax2.set_zlabel(r'$h_i$', labelpad=7)
    ax1.set_ylim((-55, -15))
    ax2.set_zlim((-55, -15))
    ax2.set_ylim((0, 1))
    ax2.view_init(azim=60, elev=15)
    print("Creating images. This may take a few seconds.")
    scas = []
    for i in range(0, int(len(pkl_data["tt"])), 50):
        sca1 = ax1.scatter(np.arange(len(pkl_data["xx"])),
                           pkl_data["data"][i, ids, 0], zorder=10, color='black', s=5)
        sca2 = ax2.scatter(v_scaled1[v_scaled2 > 0],
                           v_scaled2[v_scaled2 > 0],
                           pkl_data["data"][i, v_scaled2 > 0, 0], zorder=10, color='black', s=5)

        coeffs = R(pkl_data["data"][i, :, :], fitter)
        # coeffs = coeffs.reshape(int(coeffs.size/2), 2).T
        coeffs = coeffs.reshape(2,  int(coeffs.size/2))
        fit = BasisFit(coeffs[0, :], fitter.basis)
        Zfull = fit(Xfull, Yfull)
        surf2 = ax2.plot_surface(Xfull, Yfull, Zfull, alpha=0.5, color='blue')
        # fit_phi_r = interpolate.Rbf(v_scaled1, v_scaled2, pkl_data["data"][i, :, 0], smooth=5)
        # surf2 = ax2.plot_surface(Xfull, Yfull, fit_phi_r(Xfull, Yfull), alpha=0.5, color='blue')

        # wgrid = ax.plot_wireframe(phihat1, phihat2, data_dict["data"][i].real, rstride=2, cstride=2)

        # scas.append([sca, wgrid])
        scas.append([sca1, sca2, surf2])

    ani = animation.ArtistAnimation(
        fig, scas, interval=100, blit=True, repeat_delay=0, repeat=True)
    writergif = animation.PillowWriter(fps=20)
    ani.save(config["DATA"]["path"] + 'double_vid_surf_ord_' +
             str(maxOrd)+'_rot2.gif', writer=writergif)
    plt.show()
