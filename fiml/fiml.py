"""
Core FIML utilities

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline


def create_basis_functions(N_basis, p=3, N_t=500, PLOT=False):
    # N_basis = N_knots - p - 1
    # for open/clamped splines, the endpoints are both repeated (p+1) times in the
    # vector of knots
    # N_knots = N_int + (p+1) + (p+2) = N_int + 2p + 2
    # therefore N_int = N_knots - 2p - 2 = (N_basis + p + 1) - 2p - 2
    N_int_knots = N_basis - p - 1
    # uniformly spaced knots
    knots = np.linspace(0, 1, N_int_knots+2)
    # internal knots: points where the cubic polynomials are joined
    int_knots = knots[1:-1]
    # open/clamped splines: repeat endpoints p+1 times => 1st endpoint has 
    # spline coordinate t = 0, whereas last point has coordinate t = 1
    knots_all = np.concatenate((np.zeros(p+1),
                                int_knots,
                                np.ones(p+1)))
    B = []
    for n in range(N_basis):
        coeff = np.zeros(N_basis)
        coeff[n] = 1
        spline = BSpline(knots_all, coeff, p)
        B.append(spline)
    
    if PLOT:
        plt.figure()
        # normalised coordinate x/L in [0,1]
        t = np.linspace(0, 1, N_t)
        B_eval = np.zeros((N_basis, N_t))
        for n in range(N_basis):
            coeff = np.zeros(N_basis)
            coeff[n] = 1
            spline = B[n]
            B_eval[n, :] = spline(t)
            plt.plot(t, B_eval[n, :], label=f'$B_{n}$')
        plt.xlabel(r'$\xi$ [-]')
        plt.ylabel(r'$B_i$ [-]')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend()
        plt.title(f'{N_basis} cubic B-splines')
        plt.show()
    
    return B
