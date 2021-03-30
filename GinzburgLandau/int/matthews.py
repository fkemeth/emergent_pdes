"""Integrate heterogenesous ensemble of Stuart-Landau oscillators with linear global coupling."""

######################################################################################
#                                                                                    #
# Jan 2020                                                                           #
# felix@kemeth.de                                                                    #
#                                                                                    #
######################################################################################

import sys
from time import time
import numpy as np
import scipy.integrate as sp

#############################################################################
# INTEGRATION
#############################################################################


def create_initial_conditions(ic, N):
    """Specify initial conditions for zero-flux boundary conditions."""
    if ic == 'random':
        y0 = 0.5 * np.ones(N) + 0.3 * np.random.randn(N)
    if ic == 'randomabs':
        y0 = np.abs(1.0 * np.ones(N) + 0.3 * np.random.randn(N))
    elif ic == 'weakrandom':
        y0 = np.ones(N) + 0.01 * np.random.randn(N) + 0.01j * np.random.randn(N)
    elif ic == 'weakrandom_asynch':
        y0 = np.ones(N) + 0.01 * np.random.randn(N) + 0.01j * np.random.randn(N)
        y0[-N / 2:] = -y0[-N / 2:]
    elif ic == 'veryweakrandom':
        y0 = np.ones(N) + 0.001 * np.random.randn(N) + 0.001j * np.random.randn(N)
    elif ic == 'twophasesrandomized':
        y0 = np.ones(N)
        y0[0:N / 2] = y0[0:N / 2] - 0.3
        y0[N / 2:N] = y0[N / 2:N] + 0.3
        y0 = y0 + 0.01 * np.random.randn(N)
    elif ic == 'synchronized':
        y0 = np.ones(N)
    elif ic == 'matthews':
        y0 = np.random.uniform(-1, 1, N) + 1.0j*np.random.uniform(-1, 1, N)
    elif ic == 'felix':
        y0 = 1.0+np.random.uniform(-1, 1, N) + 1.0j*np.random.uniform(-1, 1, N)
        y0 -= np.mean(y0)+1.0
    elif ic == 'felix_gauss':
        y0 = 1.0+1e-3*np.random.randn(N) + 1e-3*1.0j*np.random.randn(N)
        y0 -= np.mean(y0)+1.0
    elif ic == 'grid':
        y0 = np.random.uniform(-1, 1, N) + 1.0j*np.random.uniform(-1, 1, N)
    return y0


def get_omega(gamma, N, random=False, gamma_off=0.):
    """Draw omegas from unimodal distribution."""
    if random:
        omega = np.random.uniform(-gamma, gamma, N)+gamma_off
    else:
        omega = np.linspace(-gamma, gamma, N)+gamma_off
    # if centered:
    #     omega[-1] = -np.sum(omega[:-1])
    return omega


def get_order_parameter(Ad):
    return np.mean(Ad["data"], axis=1)


def f(t, y, arg_pars):
    """Temporal evolution with linear global coupling."""
    glob_lin = np.mean(y) - y
    return y*(arg_pars["lambda"]+1.0j*arg_pars["omega"]-abs(y)**2)+arg_pars["K"]*glob_lin


def jac(t, y, arg_pars):
    """Calculate the jacobian evaluated at y."""
    N = y.shape[0]
    J = np.zeros((2 * N, 2 * N))
    J[:N, :N] = arg_pars["K"] / float(N)
    J[N:, N:] = arg_pars["K"] / float(N)
    np.fill_diagonal(J[:N, :N], arg_pars["lambda"]-arg_pars["K"]-np.imag(y[:]) **
                     2-3*np.real(y)**2+arg_pars["K"] / float(N))
    np.fill_diagonal(J[:N, N:], -arg_pars["omega"]-2*np.real(y)*np.imag(y))
    np.fill_diagonal(J[N:, :N], arg_pars["omega"]-2*np.real(y)*np.imag(y))
    np.fill_diagonal(J[N:, N:], arg_pars["lambda"]-arg_pars["K"]-np.real(y[:])**2-3*np.imag(y)
                     ** 2+arg_pars["K"] / float(N))
    return J


def integrate(dynamics='', pars=dict({"lambda": 1.0, "gamma": 1.0, "K": 1.0, "omega": False}),
              N=2, tmin=500, tmax=1000, dt=0.01, T=1000, ic='matthews', Ainit=0,
              gamma_off=0., append_init=False):
    """Integrate heterogeneous ensemble of Stuart-Landau oscillators with linear global coupling."""
    tstart = time()

    gamma = pars["gamma"]
    K = pars["K"]
    omega = pars["omega"]
    if "lambda" not in pars.keys():
        pars["lambda"] = 1.0
    lmbda = pars["lambda"]

    # Predefined dynamics are:
    if dynamics == 'locking':
        print("Taking parameters for " + dynamics + " dynamics.")
        gamma = 1.0
        K = 1.0
    else:
        print("No predifined dynamics selected. Taking specified parameters.")

    if omega is None:
        omega = get_omega(gamma, N, gamma_off=gamma_off)

    # Write the parameters into a dictionary for future use.
    Adict = dict()
    Adict["gamma"] = gamma
    Adict["gamma_off"] = gamma_off
    Adict["omega"] = omega
    Adict["lambda"] = lmbda
    Adict["K"] = K
    Adict["N"] = N
    Adict["tmin"] = tmin
    Adict["tmax"] = tmax
    Adict["dt"] = dt
    Adict["T"] = T
    Adict["ic"] = ic

    # Number of timesteps
    nmax = int(np.abs(float(tmax) / float(dt)))
    if T > nmax:
        raise ValueError('T is larger than the maximal number of time steps.')

    # Number of timesteps above threshold
    n_above_thresh = int(float(np.abs((tmax - tmin)) / float(dt)))
    # Threshold itself
    n_thresh = nmax - n_above_thresh
    # Every nplt'th step is plotted
    nplt = n_above_thresh / float(T)

    if ic == 'manual':
        if (Ainit.shape[0] != N):
            raise ValueError('Initial data must have the specified N dimension.')
        y0 = Ainit
    else:
        y0 = create_initial_conditions(ic, N)

    Adict["init"] = y0

    ydata = list()
    T = list()

    t0 = 0.0
    r = sp.ode(f).set_integrator('zvode', method='Adams', atol=1e-11, rtol=1e-11, nsteps=1e7)
    r.set_initial_value(y0, t0).set_f_params(pars)

    if append_init and (tmin == 0):
        ydata.append(y0)

    i = 0
    while r.successful() and np.abs(r.t) < np.abs(tmax):
        i = i + 1
        r.integrate(r.t + dt)
        if (i >= n_thresh) and (i % nplt == 0):
            T.append(r.t)
            ydata.append(r.y)

        if i % (np.floor(nmax / 10.0)) == 0:
            sys.stdout.write("\r %9.1f" % round((time() - tstart) / (float(i)) *
                                                (float(nmax) - float(i)), 1) + ' seconds left')
            sys.stdout.flush()
    print("\n")
    Adict["tt"] = np.array(T)
    Adict["xx"] = omega
    Adict["L"] = 2*gamma
    Adict["pars"] = pars
    Adict["param"] = gamma
    Adict["data"] = np.array(ydata)
    return Adict
