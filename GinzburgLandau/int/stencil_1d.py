"""Create 1d discrete Laplacian stencil."""

######################################################################################
#                                                                                    #
# Jun 2016                                                                           #
# felix@kemeth.de                                                                    #
#                                                                                    #
######################################################################################

import numpy as np
from scipy.sparse import csr_matrix


def create_stencil(N):
    """Create finite difference stencil."""
    stdata = np.empty(3 * N)
    stdata[0:N].fill(-2.0)
    idxs = np.empty(N)
    idxs[0:N] = np.arange(0, N)
    idxscol = np.empty(N)
    idxscol[0:N] = np.arange(0, N)
    idxscolrechts = np.empty(N)
    idxscolrechts = np.arange(0, N) + 1
    idxscolrechts[-1] = 0
    idxscollinks = np.empty(N)
    idxscollinks = np.arange(0, N) - 1
    idxscollinks[0] = N - 1
    stdata[N:3 * N].fill(1.0)
    idxtmp = idxs
    for x in range(0, 2):
        idxs = np.append(idxs, idxtmp)
    idxscol = np.append(idxscol, idxscolrechts)
    idxscol = np.append(idxscol, idxscollinks)
    stencil = csr_matrix((stdata, (idxs, idxscol)), shape=(N, N), dtype=float)
    return stencil
