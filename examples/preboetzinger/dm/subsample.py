__all__ = ['subsample']

import numpy as np
import scipy
from scipy.spatial import distance

def subsample(points, min_distance=None, tol=1e-4, n_samples=100, random_state=None, randomized=False, verbosity_level=2):

        """

        Returns a new set of points that has a converged subsampling of the given points.

        randomized: False (default, will subsample iteratively) True (will randomly pick indices uniformly. Very fast)

        """

        if points.shape[0] == 0:

            raise AssertionError("data set does not contain any points.")


        if not(random_state is None):

            np.random.seed(random_state)

       

        if min_distance is None:

            min_distance = 1

           

        if verbosity_level > 0:

            print('Subsampling points.')

           

        if randomized:

            new_indices_total = np.random.permutation(points.shape[0])[0:np.max([10000, int(np.sqrt(points.shape[0]))])]

            new_points = points[new_indices_total,:]

        else:

            if verbosity_level > 1:

                print('Considering min_distance %f.' % min_distance)

            

            indices = np.random.permutation(points.shape[0])

            new_indices_total = indices[0 : n_samples].tolist()

            new_points = points[new_indices_total,:].tolist()

            for k in range(1, points.shape[0]//n_samples+1):

               

                new_indices = indices[(k*n_samples) : ((k+1)*n_samples)]

                if len(new_indices) <= 1:

                    break

                new_points_k = points[new_indices,:]

                

                distances = distance.cdist(np.row_stack(new_points), new_points_k)

                distances[distances > min_distance] = 0

                distances = scipy.sparse.csr_matrix(distances.T)

               

                new_indices_k = []

                for i in range(distances.shape[0]):

                    row = distances.getrow(i)

                    if len(row.data) == 0:

                        new_indices_k.append(new_indices[i]) # consider this point

                    elif np.min(row.data) >= min_distance:

                        new_indices_k.append(new_indices[i]) # consider this point

                       

                if len(new_indices_k) < int(n_samples * tol)+1:

                    break

               

                new_points_k = points[new_indices_k,:]

               

                new_indices_total.extend(new_indices_k)

                new_points.append(new_points_k)

           

        if verbosity_level > 1:

            print('Subsampling complete, taking %g out of %g points.' % (len(new_indices_total), points.shape[0]))

            

        return np.row_stack(new_points), new_indices_total