"""Integration of the complex Ginzburg-Landau equation with 1 spatial dimension."""

#############################################################################
#                                                                           #
# Integration of the complex Ginzburg Landau eq.                            #
# with linear global coupling through the mean field.                       #
# See Master's thesis Felix Kemeth                                          #
#                                                                           #
# Mar 2016                                                                  #
# felix@kemeth.de                                                           #
#                                                                           #
#############################################################################

import numpy as np
from time import time
import sys

#############################################################################
# INTEGRATION
#############################################################################


def initial_conditions(bc, ic, L, N):
    """Specify initial conditions depending on boundary conditions."""
    Lp = 2 * L
    print("Creating initial conditions (" + ic + ") with " + bc + " boundary conditions.")
    if bc == "no-flux":
        # Create grid
        Xp = (Lp / N) * np.arange(-N / 2, N / 2)

        # Pre-defined Initial Condition (not from file)
        if ic == 'pulse':
            A = 0.5 + 0.5 * (1 / np.cosh((Xp**2) * 0.2)) + \
                10**(-2) * np.random.randn(len(Xp))
            A = A / np.average(A)
        elif ic == 'sine1D':
            A = 0.01 * np.sin(2 * np.pi * Xp / Lp) + 0.5
            A = A / np.average(A)
        elif ic == 'plain':
            A = Xp * 0.0 + 0.5
            A = A / np.average(A)
        elif ic == 'plain_random':
            A = 0.5 + 0.001 * np.random.randn(len(Xp))
            A = A / np.average(A)
        elif ic == 'random':
            A = 0.5 * np.random.randn(len(Xp))
    elif bc == "periodic":
        # Create grid
        X = (L / N) * np.arange(-N / 2, N / 2)

        # Pre-defined Initial Condition (not from file)
        if ic == 'pulse':
            A = 0.5 + 0.5 * (1 / np.cosh((X**2) * 0.2)) + \
                10**(-2) * np.random.randn(len(X))
            A = A / np.average(A)
        elif ic == 'sine1D':
            A = 0.01 * np.sin(2 * np.pi * X / L) + 0.5
            A = A / np.average(A)
        elif ic == 'plain':
            A = X * 0.0 + 0.5
            A[:N / 3] = A[:N / 3] + 0.01 * np.random.randn(N / 3)
            A = A / np.average(A)
        elif ic == 'plain_random':
            A = 0.5 + 0.001 * np.random.randn(len(X))
            A = A / np.average(A)
        elif ic == 'random':
            A = 0.5 * np.random.randn(len(X))
    else:
        print("Please set proper boundary conditions: 'periodic' or 'no-flux'")
    return A


def pad_reflect_1D(array, N):
    """Create NxN data matrix from N/2xN/2 matrix with zero flux boundaries."""
    # deltaY/deltaX should not exceed length of corresponding dimension of array.
    deltax = int(N / 4)
    output = np.zeros((len(array) + 2 * deltax)).astype(np.complex256)
    output[deltax:-deltax] = array
    output[0:deltax] = output[2 * deltax:deltax:-1]
    output[-deltax:] = output[-deltax - 2:-2 * deltax - 2:-1]
    return output.astype(np.complex128)


def integrate(dynamics='type 1', c1=0.0, c2=0.0, c3=0.0, mu=0.0,
              L=400.0, N=2**12, tmin=1000, tmax=1500, dt=0.05,
              T=1000, bc='periodic', ic='random', Ainit=0):
    """Integrate CGLE and return dictionary A."""
    tstart = time()
    L = float(L)
    # Predefined dynamics are:
    # 'type 1'                  Type I Chimera
    # 'localized turbulence'    Localized Turbulence
    # 'intermittency'           Intermittency
    # 'spirals'                 Spirals
    # 'defect'                  Defect turbulence

    print("Integrating CGLE:")
    if dynamics == 'type 1':
        print("Taking parameters for " + dynamics + " dynamics.")
        c1 = 1.2
        c2 = 1.7
        c3 = -1.25
        mu = 0.67
    elif dynamics == 'localized turbulence':
        print("Taking parameters for " + dynamics + " dynamics.")
        c1 = 2.0
        c2 = -1.2
        c3 = 0.0
        mu = 0.3
    elif dynamics == 'intermittency':
        print("Taking parameters for " + dynamics + " dynamics.")
        c1 = 0.0
        c2 = -3.0
        c3 = 0.0
        mu = 0.0
    elif dynamics == 'spirals':
        print("Taking parameters for " + dynamics + " dynamics.")
        c1 = 0.0
        c2 = 1.0
        c3 = 0.0
        mu = 0.0
    elif dynamics == 'defect':
        print("Taking parameters for " + dynamics + " dynamics.")
        c1 = -4.0
        c2 = 1.0
        c3 = 0.0
        mu = 0.0
    else:
        print("No predifined dynamics selected. Taking specified parameters.")

    # Write the parameters into a dictionary for future use.
    Adict = dict()
    Adict["c1"] = c1
    Adict["c2"] = c2
    Adict["c3"] = c3
    Adict["mu"] = mu
    Adict["L"] = L
    Adict["N"] = N
    Adict["tmin"] = tmin
    Adict["tmax"] = tmax
    Adict["dt"] = dt
    Adict["T"] = T
    Adict["ic"] = ic
    Adict["bc"] = bc

    # Simulation parameters
    nmin = 0
    # Number of timesteps
    nmax = int(tmax / dt)
    if T > nmax:
        raise ValueError('T is larger than the maximal number of time steps.')
    # Number of timesteps above threshold
    n_above_thresh = int((tmax - tmin) / dt)
    # Threshold itself
    n_thresh = nmax - n_above_thresh
    # Every nplt'th step is plotted
    nplt = n_above_thresh / T

    # data lists
    # data in physical space
    Adata = list()
    # Time Array
    T = list()

    # Set initial conditions
    if ic == 'manual':
        print("Using the given data as initial conditions.")
        A = Ainit
        if bc == 'periodic':
            if (Ainit.shape[0] != N):
                raise ValueError('Initial data must have the specified NxN dimension.')
            A_hat = np.fft.fft(A)
        elif bc == 'no-flux':
            if (Ainit.shape[0] != int(N / 2)):
                raise ValueError('Initial data must have the specified N/2xN/2 dimension.')
            # Make a NxN matrix out of the N/2xN/2 data through mirroring.
            A = pad_reflect_1D(A, N)
            A_hat = np.fft.fft(A)
            # B_nxny = (-1)**nx * A_nxny
            A_hat[int(N/2) + 1:N][::2] = -A_hat[1:int(N/2)][::-1][::2]
            # No-flux boundary conditions
            A_hat[int(N/2) + 2:N][::2] = A_hat[1:int(N/2) - 1][::-1][::2]
            A = np.fft.ifft(A_hat)
        else:
            raise ValueError("Please set proper boundary conditions: 'periodic' or 'no-flux'")
    else:
        A = initial_conditions(bc, ic, L, N)
        A_hat = np.fft.fft(A)
        if bc == 'no-flux':
            # B_nxny = (-1)**nx * A_nxny
            A_hat[int(N/2) + 1:N][::2] = -A_hat[1:int(N/2)][::-1][::2]
            # No-flux boundary conditions
            A_hat[int(N/2) + 2:N][::2] = A_hat[1:int(N/2) - 1][::-1][::2]
        elif bc == 'periodic':
            pass
        else:
            raise ValueError("Please set proper boundary conditions: 'periodic' or 'no-flux'")
        A = np.fft.ifft(A_hat)

    Adict["init"] = A

    # Set of wavenumbers
    k = np.concatenate(
        (np.arange(int(N/2) + 1), np.arange(-int(N/2) + 1, 0))) * 2 * np.pi / L
    k2 = k**2

    if tmin == 0:
        if bc == "no-flux":
            # Take only half (in each dimension) of the data points +1
            Adata.append(np.array(A[int(N/4):int(3*N/4) + 1]))
        elif bc == "periodic":
            Adata.append(A[:])
        # Here one has to save the full spectrum; first step after n_thresh
        # saved for later continuation
        T.append(0)

    # Compute exponentials and nonlinear factors for ETD2 method
    cA = 1.0 - k2 * (1.0 + c1 * 1j) - mu * (1.0 + 1j * c3)
    # Homogeneous mode
    cA[0] = 1.0
    expA = np.exp(dt * cA)
    nlfacA = (np.exp(dt * cA) * (1 + 1 / (cA * dt)) - 1 / (cA * dt) - 2) / cA
    nlfacAp = (np.exp(dt * cA) * (-1 / (cA * dt)) + 1 / (cA * dt) + 1) / cA

    for i in np.arange(nmin + 1, nmax + 1):
        # Calculation of nonlinear part in Fourier space
        nlA = -(1.0 + 1.0j * c2) * np.fft.fft(A * abs(A)**2)

        # Setting the first values of the previous nonlinear coefficients
        if i == nmin + 1:
            nlAp = nlA

        if bc == "no-flux":
            # Time-stepping (carried out in parallel for each individual Fourier mode)
            A_hat[0:int(N/2) + 1] = A_hat[0:int(N/2) + 1] * expA[0:int(N/2) + 1] + \
                nlfacA[0:int(N/2) + 1] * nlA[0:int(N/2) + 1] + nlfacAp[0:int(N/2) + 1] * \
                nlAp[0:int(N/2) + 1]
            # Increasing efficiency by invoking boundary condition.

            # No-flux boundary conditions
            A_hat[int(N/2) + 1:N][::2] = -A_hat[1:int(N/2)][::-1][::2]
            A_hat[int(N/2) + 2:N][::2] = A_hat[1:int(N/2) - 1][::-1][::2]
        elif bc == "periodic":
            # Time-stepping (carried out in parallel for each individual Fourier mode)
            A_hat[:] = A_hat[:] * expA[:] + nlfacA[:] * nlA[:] + nlfacAp[:] * nlAp[:]
        else:
            print("Please set proper boundary conditions: 'periodic' or 'no-flux'")

        A = np.fft.ifft(A_hat)
        nlAp = nlA

        # Saving data each nplt'th step
        if ((i > n_thresh) and (i % nplt == 0)):
            if bc == "no-flux":
                # again only half of the data +1
                Adata.append(A[int(N / 4):int(3*N/4 + 1)])
            elif bc == "periodic":
                Adata.append(A[:])
            T.append(i * dt)
        # Simulation time countdown
        if (i - nmin) % (np.floor((nmax - nmin) / 100)) == 0:
            sys.stdout.write("\r %9.1f" % round((time() - tstart) / (float(i) - nmin) *
                                                (float(nmax) - float(i)), 1) + ' seconds left')
            sys.stdout.flush()

    print("\n")
    tend = time()
    print('Simulation completed!')
    print('Running time: ' + str(round((tend - tstart), 1)) + 's')
    Adict["data"] = np.array(Adata)
    return Adict
