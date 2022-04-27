"""
Copyright © 2022 Felix P. Kemeth

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
######################################################################################
#                                                                                    #
# Jun 2022                                                                           #
# felix@kemeth.de                                                                    #
#                                                                                    #
######################################################################################

import numpy as np
from scipy.sparse import csr_matrix


def create_stencil(num_grid_points: int) -> csr_matrix:
    """
    Create 2nd order finite difference stencil in one dimension.

    :param num_grid_points: integer with the number of grid points of
                            spatial axis
    :returns: finite difference stencil as csr matrix
    """
    stencil_data = np.empty(3 * num_grid_points)
    stencil_data[0:num_grid_points].fill(-2.0)
    idxs = np.empty(num_grid_points)
    idxs[0:num_grid_points] = np.arange(0, num_grid_points)
    idxscol = np.empty(num_grid_points)
    idxscol[0:num_grid_points] = np.arange(0, num_grid_points)
    idxscolright = np.empty(num_grid_points)
    idxscolright = np.arange(0, num_grid_points) + 1
    idxscolright[-1] = 0
    idxscolleft = np.empty(num_grid_points)
    idxscolleft = np.arange(0, num_grid_points) - 1
    idxscolleft[0] = num_grid_points - 1
    stencil_data[num_grid_points:3 * num_grid_points].fill(1.0)
    idxtmp = idxs
    for _ in range(0, 2):
        idxs = np.append(idxs, idxtmp)
    idxscol = np.append(idxscol, idxscolright)
    idxscol = np.append(idxscol, idxscolleft)
    stencil = csr_matrix(
        (stencil_data, (idxs, idxscol)),
        shape=(num_grid_points, num_grid_points),
        dtype=float,
    )
    return stencil
