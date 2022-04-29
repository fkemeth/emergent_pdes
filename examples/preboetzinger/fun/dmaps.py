"""Diffusion maps functions."""

###############################################################################
#                                                                             #
# doi: 10.1073/pnas.0500334102                                                #
#                                                                             #
# Jun 2016                                                                    #
# felix@kemeth.de                                                             #
#                                                                             #
###############################################################################

import numpy as np
import time
import scipy.spatial as sp
import scipy.sparse.linalg as spsp


def dmaps(A, evals=20, eps=False, alpha=0):
    """Apply diffusion maps with Euklidean distance and median scale."""
    print('applying dmaps...')
    stamp = time.time()
    T, N = A.shape

    # (Squares of) pairwise Euclidian distances between all timepoints 1,..,T:
    Adist = np.square(sp.distance.pdist(A[:, :].real, 'euclidean')) + \
        np.square(sp.distance.pdist(A[:, :].imag, 'euclidean'))

    # Kernel function:
    if not eps:
        eps = np.median(Adist)
        print('Using median epsilon of: '+str(eps)+' or ln: '+str(np.log(eps)))
    K = np.exp(-sp.distance.squareform(Adist, force='tomatrix') / (2.0 * eps))

    if alpha > 0:
        diag = np.sum(K, axis=1)
        inv_diag = 1.0/diag**alpha
        Dinv = np.diag(inv_diag)
        K = np.dot(Dinv, np.dot(K, Dinv))

    # Create size-TxT diffusion (probability) matrix by normalizing row-wise:
    for i in np.arange(0, K.shape[0]):
        K[i, :] = K[i, :] / np.sum(K[i, :])

    # Obtain largest eigenvalues and corr. eigenvectors:
    if (evals > T - 2):
        print("evals >= N-2 -> Taking numpy eigenvalue solve.")
        D, V = np.linalg.eig(K)
    else:
        D, V = spsp.eigs(K, evals, maxiter=10000)
    idx = np.abs(D).argsort()[::-1]
    D = D[idx]
    V = V[:, idx]

    # Check if dmaps outcome is accurate
    if np.any(V.imag) > 1e-14 or np.any(D.imag) > 1e-14:
        raise ValueError("Error: Dmaps calculation not accurate!")

    print("--- %s seconds ---" % (time.time() - stamp))
    return D.real, V.real
