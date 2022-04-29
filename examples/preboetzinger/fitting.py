'''
Created on Aug 8, 2014
@author: bertalan@princeton.edu
'''
import numpy as np
import logging


class F(object):
    """Product function."""

    def __init__(self, combo):
        self.combo = combo

    def __call__(self, *args):
        out = 1
        for g, arg in zip(self.combo, args):
            if isinstance(g, Basis):
                pass  # TODO: Wait, what?
            out *= g(arg)
        return out

    def __repr__(self):
        return "product of functions: %s" % ' * '.join(['('+str(g)+')' for g in self.combo])


def product(maxOrder, *bases):
    """Generate a Cartesian product basis from the given bases.
    If maxOrder : int is given, f.order will be checked for each element in the
    basis, to check that the sum of orders for any given combination from the
    Cartesian product is <= maxOrder. Otherwise, all combinations are used.
    >>> from pbPce.basisFit.polynomials import Basis, Polynomial1D, showBasis, StdBasis
    >>> basis = product(3, StdBasis(2), StdBasis(1))
    >>> print showBasis(basis, 2)
    1
    x1
    x0
    x0*x1
    x0**2
    x0**2*x1
    >>> b1 = Basis([Polynomial1D([1]), Polynomial1D([0, 1])]); b1
    Basis of 2 functions:
        1D Polynomial: 1x**0
        1D Polynomial: 1x**1 + 0x**0
    >>> b2 = Basis([Polynomial1D([3]), Polynomial1D([5, 3, 9, 4]), Polynomial1D([5])]); b2
    Basis of 3 functions:
        1D Polynomial: 3x**0
        1D Polynomial: 4x**3 + 9x**2 + 3x**1 + 5x**0
        1D Polynomial: 5x**0
    >>> basis = product(3, b1, b2)
    >>> print showBasis(basis, 2)
    3
    4*x1**3 + 9*x1**2 + 3*x1 + 5
    5
    3*x0
    5*x0
    """
    from itertools import product
    combos = []

    logging.getLogger(__name__).debug('maxOrder = %d' % maxOrder)
    logging.getLogger(__name__).debug('%d combinations:' %
                                      len(list(product(*[list(b) for b in bases]))))
    for combo in product(*[list(b) for b in bases]):
        from sympy import symbols
        xl = symbols(['x%d' % i for i in range(len(bases))])
        totOrder = sum([f.order for f in combo])
        f = F(combo)
        fstr = str(f(*xl)) + " (order %d)" % totOrder
        if maxOrder is None or totOrder <= maxOrder:
            for g in combo:
                assert callable(g)
            logging.getLogger(__name__).debug('accepting %s' % fstr)
            combos.append(combo)
        else:
            logging.getLogger(__name__).debug("rejecting %s" % fstr)
    return [F(combo) for combo in combos]


class Polynomial1D(object):
    """A 1D polynomial constructed from the standard monomial basis.
    >>> p = Polynomial1D((1, 2, 3))
    >>> p
    1D Polynomial: 3x**2 + 2x**1 + 1x**0
    >>> p(5) == 1 + 2*5 + 3*5**2
    True
    >>> p.order
    2
    """
    # TODO : basisFit.polynomials.discrete.discretePolynomial needs to be merged into here.

    def __init__(self, stdCoeffs):
        self.order = len(stdCoeffs) - 1
        assert self.order >= 0
        self._stdCoeffs = stdCoeffs
        self.p = len(stdCoeffs) - 1
        self._stdBasis = StdBasis(self.p)

    @property
    def coeffs(self):
        return self._stdCoeffs

    @coeffs.setter
    def coeffs(self, coeffs):
        self._stdCoeffs = coeffs

    def __call__(self, arg):
        return sum([c*f(arg) for c, f in zip(self._stdCoeffs, self._stdBasis)])

    def __repr__(self):
        return "%s: %s" % ('1D Polynomial',
                           ' + '.join(['%sx**%d' % (c, self.p - i)
                                       for (i, c) in enumerate(self._stdCoeffs[::-1])]
                                      )
                           )


def polynomialsFromMoments(X, maxOrd):
    """
    Oladyshkin, S. & Nowak, W. Data-driven uncertainty quantification using the
    arbitrary polynomial chaos expansion. Reliab. Eng. Syst. Saf. 106, (2012).
    >>> N = 100000; X = np.random.normal(loc=0, scale=1, size=(N,))
    >>> polys = polynomialsFromMoments(X, 4)
    Compare to a Hermite basis:
    >>> from hermite import HndBasis; hpolys = HndBasis(4, 1)
    >>> from sympy.abc import x; from sympy import expr
    >>> for p in hpolys: print expr.Expr(p(x)).expand()
    Expr(1)
    Expr(x)
    Expr(x**2 - 1)
    Expr(x**3 - 3*x)
    Expr(x**4 - 6*x**2 + 3)
    notebook 20130318, pp 39-40
    """
    p00 = 1.0
    coeffs = [[p00]]
    momentMatrix = np.empty((maxOrd+1, maxOrd+1))

    mean = np.mean(X)
    N = np.size(X)

    def m(d):
        return np.sum((X - mean)**d) / (N - 1)

    for i in range(maxOrd+1):
        for j in range(maxOrd+1):
            momentMatrix[i, j] = m(i+j)

    for k in range(1, maxOrd+1):
        A = np.vstack((momentMatrix[0:k, 0:k+1], np.zeros((1, k+1))))
        b = np.zeros((k+1, 1))
        A[k, k] = b[k, 0] = 1
        u, residuals, rank, singularValues = np.linalg.lstsq(A, b)
        coeffs.append(u.ravel().tolist())

    return [Polynomial1D(c) for c in coeffs]


def generate(X, maxOrd, method="moments"):
    """Generate 1D polynomials orthonormal WRT sampled real abscissae.
    """
    method = method.lower()
    if method == "moments":
        return polynomialsFromMoments(X, maxOrd)
    else:
        raise NotImplementedError


class Basis(object):
    """A collection of callable Functions, which may or may not have other nice features.
    >>> b = Basis([Polynomial1D((1,)), Polynomial1D((-3, 0, 0, 3))])
    >>> b
    Basis of 2 functions:
        1D Polynomial: 1x**0
        1D Polynomial: 3x**3 + 0x**2 + 0x**1 + -3x**0
    """

    def __init__(self, basisList, pdf=None):
        self._basisList = basisList
        self.pdf = pdf

    def recursionFormula(self):
        raise NotImplementedError

    def __getitem__(self, sl):
        return self._basisList[sl]

    def __len__(self):
        return len(self._basisList)

    def __setitem__(self, key, value):
        # TODO: Actually, why would I ever want this?
        self._basisList[key] = value

    def __repr__(self):
        typeStr = "Basis of %d functions" % len(self._basisList)
        return "%s:\n    %s" % (typeStr, str('\n    '.join(
            [repr(f) for f in self._basisList]
        )))


def showBasis(basis, arity):
    """A human-readable depiction of a basis."""
    # TODO : This should probably be the __str__ for the Basis class.
    from sympy import symbols
    args = symbols(['x%d' % i for i in range(arity)])
    out = []
    for f in basis:
        assert callable(f)
        out.append(str(f(*args)))
    return '\n'.join(out)


class Monomial:
    def __init__(self, order):
        assert order >= 0
        self.order = order

    def __call__(self, x):
        return x ** self.order

    def __repr__(self):
        return "monomial x^%d" % self.order


class StdBasis(Basis):
    def __init__(self, maxOrder):
        """A lightweight (?) basis of standard monomials
        (that know their own orders).
        >>> basis = StdBasis(3)
        >>> from sympy import Symbol; x = Symbol('x')
        >>> [f(x) for f in basis]
        [1, x, x**2, x**3]
        >>> basis[-1].order
        3
        """
        super(StdBasis, self).__init__([Monomial(i) for i in range(maxOrder + 1)])

    def _str__(self):
        return showBasis(self, 1)


class ContinuousBasis(Basis):
    """Generate basis functions from data.
    Parameters
    ==========
    X : array_like (n, m)
        Array of abscissae, sampled from the weighted space in which the
        generated polynomials should be orthonormal.
        The method used to generate the polynomials will depend on whether the
        dtype of the array (or type of the first element) is int-like or
        float-like.
    maxOrd : int
        The maximum total order allowed in the generated polynomials.
    Returns
    =======
    basis : basis object
    Demo:
    >>> X = np.random.normal(size=(1000,)).reshape((1000,1))
    >>> b = ContinuousBasis(X, 4)
    """

    def __init__(self, X, maxOrd, assumeIndependent=True):
        self.maxOrd = maxOrd
        assert len(X.shape) == 2, X.shape
        ndims = min(X.shape)
        assert ndims == X.shape[1], "The shape of the data must start with ndims (%d)." % ndims
        def oneDeeBasis(y): return generate(y, maxOrd, method="moments")
        if ndims > 1:
            if assumeIndependent:
                listOfFunctions = [oneDeeBasis(X[:, i]) for i in range(ndims)]
                listOfFunctions = product(maxOrd, *listOfFunctions)
            else:
                raise NotImplementedError
        else:
            listOfFunctions = oneDeeBasis(X)
        super(ContinuousBasis, self).__init__(listOfFunctions)


def vandermonde(basis, explanMat):
    """Construct a pseudo-Vandermonde matrix in the given basis.
    Parameters
    ----------
    basis : list of callble basis functions of arity nvars
    explanMat : (ndata, nvars) array_like
        The explanatory data. Rows are observance instances; columns are the
        multiple variables. If this were unidimensional, there would be 1 column.
    Returns
    -------
    V : numpy.ndarray, shape=(ndata, len(basis))
    >>> from pbPce.basisFit.polynomials.hermite import HndBasis
    >>> vandermonde(HndBasis(2, 3), np.random.random((32, 3))).shape
    (32, 10)
    """
    # TODO: Make this into a class version of this, with __getitem__ etc. methods, to make larger problems possible without using all RAM.
    if len(explanMat.shape) == 1:
        explanMat = explanMat.reshape(explanMat.size, 1)
    basisSize = len(basis)
    numDatums = explanMat.shape[0]
    V = np.empty((numDatums, basisSize))
    for i in range(numDatums):
        for j, f in enumerate(basis):
            V[i, j] = f(*np.array(explanMat[i, :]).ravel())
    return V


class BasisFit(object):
    """An expansion in a basis with the given coefficients.
    >>> fit = BasisFit([4, 7], [lambda x: 3*x, lambda x: 15*x**2])
    >>> fit.coeffs
    [4, 7]
    >>> fit.coeffs = [6, 13]
    >>> fit._coeffs
    [6, 13]
    >>> fit(11) == 6*(3*11) + 13*(15*11**2)
    True
    And tested with an actual Basis:
    >>> from pbPce.basisFit.polynomials import Basis, Polynomial1D
    >>> fit = BasisFit([6, 13], Basis([Polynomial1D((0,3)), Polynomial1D((0,0,15))]))
    >>> fit(11) == 6*(3*11) + 13*(15*11**2)
    True
    """

    def __init__(self, coeffs, basis):
        self._coeffs = coeffs
        self.basis = basis

    def _getCoeffs(self):
        return self._coeffs

    def _setCoeffs(self, coeffs):
        self._coeffs = coeffs
    coeffs = property(_getCoeffs, _setCoeffs)

    def __call__(self, *args):
        return sum([c*f(*args) for (c, f) in zip(self._coeffs, self.basis)])


class BasisFitter(object):
    """Save the work of re-evaluating the Vandermonde matrix when fitting a basis.
    Since we're usually interested in cases where the abscissae are fixed (and
    therefore so is the appropriate orthonormal basis), while only the ordinates
    change (and therefore so do the fitting coefficients), we can evaluate the
    basis at the absicssae once and store that.
    """

    def __init__(self, X, maxOrd=None, basis=None, fitMethod='lsq'):
        """
        Parameters
        ==========
        X : (ndata, nvars) array_like
        maxOrd : int, optional
        basis : collection of basis functions
        fitMethod : str in ['lsq']
        """
        # TODO : Abscissae should be passed in and stored here, and basis should be optional (normally generated from the abscissae).
        if len(X.shape) == 1:
            # Ensure X is 2D, with second index across heterogeneities.
            X = X.reshape(X.size, 1)
        self.X = X

        if basis is None:
            if maxOrd is None:
                # If basis is not None, then presumably it handles maxOrd itself.
                maxOrd = 3
            basis = ContinuousBasis(X, maxOrd)

        assert isinstance(basis, Basis), type(basis)
        self.basis = basis

        self._fitMethod = fitMethod
        self.V = vandermonde(self.basis, self.X)
        self.norms = np.sqrt(np.square(self.V.T).sum(1))
        self.Vnorms = self.V / self.norms

    def fit(self, y):
        # TODO : This needs to go to the fitting module.
        if self._fitMethod.lower() == 'lsq':

            co, resid, rank, sing = np.linalg.lstsq(self.Vnorms, y)
            co = (co.T / self.norms).T  # Normalizing is important to keep the inversion here
            # well-conditioned if, for example, p and X.max()/X.min()
        else:
            raise NotImplementedError

        return BasisFit(co, self.basis)

    def evaluate(self, coeffs):
        return np.dot(self.V, coeffs)
