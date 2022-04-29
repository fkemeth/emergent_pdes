import numpy as np
from scipy.integrate import solve_ivp

import sys
if sys.version_info > (3, 0):
    basestring = str


class TwoCondEqnSim:
    gsyn = 0.3
    gl = 2.4
    Vl = -65
    # C = 0.21
    # eps = 0.1
    varNames = 'V', 'h'

    def __init__(self, N=12, Iapp=None, A=None, Vsyn=0, VNa=50, C=0.21, eps=0.1):
        '''Contain all fine simulation details in an object.
        Parameters
        ==========
        N : int, optional
            Number of cells.
        Iapp : float or array-like with N elements or string, optional
            Applied current. Defaults to uniform distribution.
            If a string is given, it should be the path to a .npz file with the
            array saved in the key 'Iapp'.
        A : array-like with shape (N,N) or string, optional
            Adjacency matrix. Defaults to all-ones.
            If a string is given, it should be the path to a .npz file with the
            array saved in the key 'A'.
        Vsyn : float, optional
        VNa : float, optional

        '''
        self.dim = 2 * N
        self.N = N
        self.nvars = 2

        if A is None:
            A = np.ones((N, N))
        if isinstance(A, basestring):
            A = np.load(A)['A']

        self.histories = [np.array([]).reshape((0, self.nvars*N)), np.array([])]

        if Iapp is None:
            Iapp = np.random.uniform(low=17.5, high=32.5, size=(N,))
        if isinstance(Iapp, basestring):
            Iapp = np.load(Iapp)['Iapp']

        self.Iapp = Iapp
        self.Vsyn = Vsyn
        self.VNa = VNa
        self.A = A
        self.C = C
        self.eps = eps
        self.gNai = np.mean([2.55, 3.05])
        self.degrees = A.sum(axis=1)

    def diff(self, t, X):
        '''V h'''
        V = X[:self.N]
        h = X[self.N:]

        gsyn = self.gsyn
        gl = self.gl
        Vl = self.Vl
        C = self.C
        eps = self.eps

        soV = 1 / (1 + np.exp(-(V + 40) / 5))
        moV = 1 / (1 + np.exp(-(V + 37) / 6))

        # TODO: Check why this doesn't match the ANOVA paper.
        tauoV = 1 / (eps * np.cosh((V + 44) / 12))

        hinfoV = 1 / (1 + np.exp((V + 44) / 6))

        Isyn = np.dot(self.A, soV) * gsyn * (self.Vsyn - V) / self.N

        dVdt = (
            -self.gNai * moV * h * (V - self.VNa)
            - gl * (V - Vl)
            + Isyn + self.Iapp
        ) / C

        dhdt = (hinfoV - h) / tauoV

#        self.tauoV_debug.append(tauoV)
#        self.V_debug.append(V)
#        self.h_debug.append(h)
#        self.dhdt_debug.append(dhdt)
#        self.Isyn_debug.append(Isyn)
#        self.hinfoV_debug.append(hinfoV)
        return np.hstack((dVdt, dhdt)).reshape(X.shape)

    def integrate(self,
                  X0=None,
                  t_span=[0, 1],
                  method='RK45',
                  t_eval=None,
                  dense_output=False,
                  events=None,
                  seed=None,
                  rtol=1e-3,
                  atol=1e-6,
                  **kwargs):
        if isinstance(X0, basestring):
            if X0.lower() == "random":
                if seed != None:
                    np.random.seed(seed)
                X0 = np.random.random((self.nvars * self.N,))
                X0[:self.N] = 100.0 * np.random.random((self.N,)) - 50.0
            else:
                X0 = np.zeros((self.nvars * self.N,))
        elif hasattr(X0, '__len__') and len(X0) == self.nvars * self.N:
            pass
        elif X0 is None:
            X0 = np.zeros((self.nvars * self.N,))
            X0[:self.N] = -60.0

        def _diff(t, X):
            return self.diff(t, X)

        self.history = solve_ivp(
            _diff,
            t_span=t_span,
            y0=X0,
            method=method,
            t_eval=t_eval,
            dense_output=dense_output,
            events=events,
            rtol=rtol,
            atol=atol,
            **kwargs
        )
        return self.history

    def getVarHists(self):
        '''
        returns history of V, h,
        '''
        assert len(self.history.t) > 0
        out = []
        # There are 2 hstack'd dynamical variables.
        for i in range(2):
            out.append(self.history.y[i*self.N:(i+1)*self.N, :])
        return out

    def getVarHistDict(self):
        '''
        >>> sim = HodgkinHuxleySim(N=4, nstep=100)
        >>> varDict = sim.getVarHistDict()
        >>> sorted(varDict.keys())
        ['V', 'h']
        >>> V = varDict['V']
        >>> V.shape
        (100, 4)
        '''
        V, h = self.getVarHists()
        return {'V': V, 'h': h}


class DataGenerator:

    def __init__(self, N=1, rtol=1e-3, atol=1e-5, **kw):
        self.simulator = TwoCondEqnSim(N=N, **kw)
        self.simulator.integrate(X0=np.reshape(
            [-60, 0.0] * N, (N, 2)).T.ravel(), t_span=[0, 2000], rtol=rtol, atol=atol)
        Vtypical, htypical = self.simulator.getVarHists()
        self.Vtypical = Vtypical[:, -1]
        self.htypical = htypical[:, -1]
        self.rawProb = .5
        self.dt = .01
        self.X0method = 'random'

    def getRandomInBasin(self, Vscale=5, hscale=.05):
        return np.hstack([
            self.Vtypical + np.random.normal(loc=0, scale=Vscale,
                                             size=(self.simulator.N,)),
            self.htypical + np.random.normal(loc=0, scale=hscale,
                                             size=(self.simulator.N,)),
        ])

    def simulate(self, tmax=None, X0='random', integrate_kw={}, **inBasinKwargs):
        if tmax is None:
            tmax = self.dt
        simulator = self.simulator
        if isinstance(X0, basestring):
            if X0 == 'widerandom':
                X0 = np.hstack([
                    np.random.uniform(low=-100, high=15, size=(simulator.N,)),
                    np.random.uniform(low=0, high=1., size=(simulator.N,)),
                ])
            elif X0 == 'random':
                X0 = np.hstack([
                    np.random.uniform(low=-60, high=-5, size=(simulator.N,)),
                    np.random.uniform(low=0, high=1, size=(simulator.N,)),
                ])
            elif X0 == 'randomInBasin':
                X0 = self.getRandomInBasin(**inBasinKwargs)
        return simulator.integrate(X0=X0, t_span=[0, tmax], **integrate_kw)

    def plotTraj(self, X, T, fig=None, axes=None, vcolor='black', hcolor='red',
                 **kwargs):
        simulator = self.simulator
        V, h = simulator.getVarHists(X)
        if fig is None:
            assert axes is None
            fig, ax = plt.subplots(figsize=(12, 6))
            ax2 = ax.twinx()
            axes = ax, ax2
        ax, ax2 = axes
        ax.plot(T, V, color=vcolor, **kwargs)

        ax2.plot(T, h, color=hcolor, **kwargs)
        ax.set_xlabel('$t$')
        ax.set_ylabel(r'$V_i(t)$')
        ax2.set_ylabel(r'$h_i(t)$', color=hcolor)
        ax2.grid(False)
        ax.grid(False)
        return fig, [ax, ax2]

    def plot(self, tmax, X0, **kwargs):
        simulator = self.simulator
        X, T = simulate(tmax=tmax, X0=X0)
        return plotTraj(X, T, **kwargs)

    def _doWork(self, seed):
        np.random.seed(seed)
        if np.random.uniform() < self.rawProb:
            # Take points in the full state space.
            states, times = self.simulate(X0=self.X0method)
        else:
            # Take points near the stable limit cycle.
            states, times = self.simulate(tmax=3, X0=self.X0method)
            print(states.shape)
            states, times = self.simulate(
                X0=states[-1]+np.random.normal(size=(2,))*np.array([.1, .01]))
        return states[0], states[-1]

    def generate(self, nsamp, numProcs=None):
        from systemidentification import procPool
        work = [(self._doWork, (seed,), {}) for seed in range(nsamp)]
        pool = procPool.Pool(work, numProcs=numProcs)
        pool.startWorking()
        pool.progressBarBlock()
        X = np.empty((nsamp, self.simulator.dim))
        Y = np.empty((nsamp, self.simulator.dim))
        for samp, result in enumerate(pool.results):
            X[samp, :], Y[samp, :] = result[1]
        return X, Y
