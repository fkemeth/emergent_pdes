"""Initial plot for paper."""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import int.matthews as mint
import fun.dmaps as dmaps

from mayavi import mlab

POINTS_W = 397.48499

plt.set_cmap('plasma')


def plot_1and2():
    N = 256
    pars = {"gamma": 1.7, "K": 1.2}
    gamma_off = 0.2
    pars["omega"] = np.linspace(-pars["gamma"], pars["gamma"], N)+gamma_off
    np.random.shuffle(pars["omega"])

    y0 = np.random.uniform(-.4, .4, N) + 1.0j*np.random.uniform(-.4, .4, N)
    Ad = mint.integrate(tmin=0, tmax=80, T=2000, ic='manual', pars=pars, N=N, Ainit=y0,
                        append_init=True)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(Ad["init"].real, Ad["init"].imag, c=Ad["init"].real)
    # ax.set_xlabel('Re W')
    # ax.set_ylabel('Im W')
    # # plt.savefig('fig/paper/plot_10.png', dpi=400)
    # plt.show()

    y0 = np.linspace(-.4, .4, int(np.sqrt(N)))
    y1 = np.linspace(-.4, .4, int(np.sqrt(N)))
    y0, y1 = np.meshgrid(y0, y1)
    Ad = mint.integrate(tmin=0, tmax=80, T=2000, ic='manual', pars=pars, N=N,
                        Ainit=y0.flatten() + 1.0j*y1.flatten(),
                        append_init=True)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(Ad["init"].real, Ad["init"].imag, c=np.arange(N))
    # ax.set_xlabel('Re W', fontsize=20, family='sans-serif')
    # ax.set_ylabel('Im W', fontsize=20, family='sans-serif')
    # ax.set_xticks((-0.4, 0., 0.4))
    # ax.set_xticklabels((-0.4, 0., 0.4), family='sans-serif', fontsize=16)
    # ax.set_yticks((-0.4, 0., 0.4))
    # ax.set_yticklabels((-0.4, 0., 0.4), family='sans-serif', fontsize=16)
    # plt.subplots_adjust(bottom=0.15)
    # # plt.savefig('fig/paper/plot_10.png', dpi=400)
    # plt.show()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    cmap = plt.get_cmap('plasma')
    cmcolors = cmap((Ad["data"][0].real-min(Ad["data"][0].real)) /
                    (max(Ad["data"][0].real)-min(Ad["data"][0].real)))
    cmcolors = cmap(np.linspace(0, 1, N))
    f = mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1), size=(3*POINTS_W, 3*POINTS_W))
    for i in range(Ad["N"]):
        mlab.plot3d(Ad["data"][:1000, i].real, Ad["data"][:1000, i].imag, Ad["tt"][:1000]/40,
                    tube_radius=0.005, colormap='Spectral', figure=f,
                    color=tuple(cmcolors[i, :-1]))
    ax = mlab.axes()
    ax.axes.label_format = '%.1f'
    ax.axes.corner_offset = 0.1
    mlab.axes(xlabel='Re W', ylabel='Im W', zlabel='t', figure=f, extent=[-.4, .4, -.4, .4, 0, 1],
              nb_labels=3, ranges=[-.4, .4, -.4, .4, Ad["tt"][0], Ad["tt"][1000]])
    mlab.points3d(Ad["data"][1000, :].real, Ad["data"][1000, :].imag, np.repeat(Ad["tt"][1000]/40, N),
                  color=(0, 0, 0),
                  figure=f, reset_zoom=False)
    mlab.view(azimuth=240, elevation=60, distance=5)
    mlab.savefig('plot_1b.png', magnification=1)
    # mlab.process_ui_events()
    # mlab_image = mlab.screenshot(figure=f)
    mlab.close()
    # mlab.show()

    tmin = 800
    f = mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1), size=(3*POINTS_W, 3*POINTS_W))
    for i in range(Ad["N"]):
        mlab.plot3d(Ad["data"][tmin:1000, i].real, Ad["data"][tmin:1000, i].imag,
                    Ad["tt"][tmin:1000]/40,
                    tube_radius=0.005, colormap='Spectral', figure=f,
                    # color=mpl.colors.to_rgb(colors[np.mod(i, len(colors))]))
                    color=tuple(cmcolors[i, :-1]))
    ax = mlab.axes()
    ax.axes.label_format = '%.1f'
    ax.axes.corner_offset = 0.1
    # ax.axes.fontsize = 8
    mlab.axes(xlabel='Re W', ylabel='Im W', zlabel='t', figure=f,
              extent=[-.4/4, .4/4, -.4/4, .4/4, Ad["tt"][tmin]/40, 1],
              nb_labels=3, ranges=[-.4/4, .4/4, -.4/4, .4/4, Ad["tt"][tmin], Ad["tt"][1000]])
    mlab.points3d(Ad["data"][1000, :].real, Ad["data"][1000, :].imag, np.repeat(Ad["tt"][1000]/40, N),
                  color=(0, 0, 0),
                  figure=f)
    cmap2 = plt.get_cmap('jet')
    cmcolors2 = cmap2((Ad["omega"]-min(Ad["omega"]))/(max(Ad["omega"])-min(Ad["omega"])))

    nodes = mlab.points3d(Ad["data"][1000].real, Ad["data"][1000].imag,
                          np.repeat(Ad["tt"][1000]/40, N)+.05,
                          (np.arctan(np.pi*(Ad["omega"]-min(Ad["omega"])) /
                                     (max(Ad["omega"])-min(Ad["omega"]))-np.pi/2)+1)/2,
                          figure=f, colormap='jet', scale_mode='none')

    mlab.view(azimuth=240, elevation=60, distance=1)
    # mlab.savefig('fig/paper/plot_1b.png', size=(POINTS_W/72, 0.4*POINTS_W/72), magnification=1)
    mlab.savefig('plot_1c.png', magnification=1)
    mlab.show()

    tmin = 800
    cmap2 = plt.get_cmap('jet')
    cmcolors2 = cmap2((np.arctan(np.pi*(Ad["omega"]-min(Ad["omega"])) /
                                 (max(Ad["omega"])-min(Ad["omega"]))-np.pi/2)+1)/2)
    # (Ad["omega"]-min(Ad["omega"]))/(max(Ad["omega"])-min(Ad["omega"])))

    f = mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1), size=(3*POINTS_W, 3*POINTS_W))
    for i in range(Ad["N"]):
        mlab.plot3d(Ad["data"][tmin:1000, i].real, Ad["data"][tmin:1000, i].imag,
                    Ad["tt"][tmin:1000]/40,
                    tube_radius=0.005, colormap='jet', figure=f,
                    # color=mpl.colors.to_rgb(colors[np.mod(i, len(colors))]))
                    color=tuple(cmcolors2[i, :-1]))
    ax = mlab.axes()
    ax.axes.label_format = '%.1f'
    ax.axes.corner_offset = 0.1
    # ax.axes.fontsize = 8
    mlab.axes(xlabel='Re W', ylabel='Im W', zlabel='t', figure=f,
              extent=[-.4/4, .4/4, -.4/4, .4/4, Ad["tt"][tmin]/40, 1],
              nb_labels=3, ranges=[-.4/4, .4/4, -.4/4, .4/4, Ad["tt"][tmin], Ad["tt"][1000]])
    mlab.points3d(Ad["data"][1000, :].real, Ad["data"][1000, :].imag, np.repeat(Ad["tt"][1000]/40, N),
                  color=(0, 0, 0),
                  figure=f)
    mlab.view(azimuth=240, elevation=60, distance=1)
    # mlab.savefig('fig/paper/plot_1b.png', size=(POINTS_W/72, 0.4*POINTS_W/72), magnification=1)
    mlab.savefig('plot_1d.png', magnification=1)
    mlab.show()

    # idxs = np.argsort(y0.real)
    idxs = np.arange(N)

    D, V = dmaps.dmaps(Ad["data"][1000:].T, eps=1e-2, alpha=1)

    V[:, 1] = 2*(V[:, 1]-np.min(V[:, 1])) / \
        (np.max(V[:, 1])-np.min(V[:, 1]))-1.

    tmin = 500

    import matplotlib.image as mpimg
    mlab_image = mpimg.imread(r'plot_1b.png')
    mlab_image2 = mpimg.imread(r'plot_1c.png')
    mlab_image3 = mpimg.imread(r'plot_1d.png')

    POINTS_W = 397.48499
    fig = plt.figure(figsize=(POINTS_W/72, 1.2*5.5))
    ax1 = fig.add_subplot(321)
    ax1.scatter(Ad["init"].real, Ad["init"].imag, c=np.arange(N), s=2)
    ax1.set_xlabel('Re W')
    ax1.set_ylabel('Im W')
    ax1.set_xticks((-0.4, 0., 0.4))
    ax1.set_xticklabels((-0.4, 0., 0.4))
    ax1.set_yticks((-0.4, 0., 0.4))
    ax1.set_yticklabels((-0.4, 0., 0.4))
    ax2 = fig.add_subplot(322)
    crop = 180
    ax2.imshow(mlab_image[int(1.5*crop):-crop, int(1.5*crop):-int(2*crop)])
    ax2.set_axis_off()
    ax3 = fig.add_subplot(323)
    crop = 100
    ax3.imshow(mlab_image2[crop:-crop, crop:-crop])
    ax3.set_axis_off()
    ax4 = fig.add_subplot(324)
    crop = 100
    ax4.imshow(mlab_image3[crop:-crop, crop:-crop])
    ax4.set_axis_off()

    ax5 = fig.add_subplot(325, projection='3d')
    for i in range(int(Ad["data"].shape[1])):
        ax5.plot(np.repeat(np.arange(len(idxs))[i], len(Ad["tt"])-tmin), Ad["tt"][tmin:],
                 Ad["data"][1+tmin:, idxs[i]].real, lw=0.09, color='k')
    ax5.set_xlabel(r'$i$', labelpad=2)
    ax5.set_ylabel(r'$t$')
    ax5.set_zlabel(r'Re W', labelpad=6)
    ax6 = fig.add_subplot(326, projection='3d')
    for i in range(int(Ad["data"].shape[1])):
        ax6.plot(np.repeat(V[i, 1], len(Ad["tt"])-tmin), Ad["tt"][tmin:],
                 Ad["data"][1+tmin:, i].real, lw=0.09, color='k')
    ax6.set_xlabel(r'$\phi_i$', labelpad=2)
    ax6.set_ylabel(r'$t$')
    ax6.set_zlabel(r'Re W', labelpad=6)
    plt.subplots_adjust(top=0.99, wspace=0.2, right=0.95, bottom=0.05, left=0.12, hspace=0.2)
    ax5.view_init(elev=60., azim=10)
    ax6.view_init(elev=60., azim=10)
    ax1.text(-0.25, 0.95, r'$\mathbf{a}$', transform=ax1.transAxes, weight='bold', fontsize=12)
    ax2.text(-0.1,  0.95, r'$\mathbf{b}$', transform=ax2.transAxes, weight='bold', fontsize=12)
    ax3.text(-0.23, 1., r'$\mathbf{c}$', transform=ax3.transAxes, weight='bold', fontsize=12)
    ax4.text(-0.05,  1., r'$\mathbf{d}$', transform=ax4.transAxes, weight='bold', fontsize=12)
    ax5.text(40.0, -1.0, 2.0, r'$\mathbf{e}$', transform=ax5.transAxes, weight='bold', fontsize=12)
    ax6.text(-9,  1.2, 1.8, r'$\mathbf{f}$', transform=ax6.transAxes, weight='bold', fontsize=12)
    plt.savefig('figure_1.pdf', dpi=400)
    plt.savefig('figure_1.png', dpi=400)
    plt.show()

    np.save('Source_Data_Figure_1.npy', Ad["data"])
