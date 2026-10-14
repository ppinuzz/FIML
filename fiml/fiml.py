"""
Core FIML utilities.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""


import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
#from functools import partial

# no need to JIT compile it, as this function is called only once during the
# optimisation & you'd need to pass p as a static argument, as it's used to
# determine an array shape, which must be known at compile-time
#@jax.jit
def create_basis_functions_matrix(N_basis, xi, p=3, PLOT=False, domain=None):
    if N_basis < p+1:
        raise ValueError('The number of basis functions must be at least p+1, '
                         f' with p being the order of the spline (here: p = {p})')
    # N_basis = N_knots - p - 1
    # for open/clamped splines, the endpoints are both repeated (p+1) times in the
    # vector of knots
    # N_knots = N_int + (p+1) + (p+2) = N_int + 2p + 2
    # therefore N_int = N_knots - 2p - 2 = (N_basis + p + 1) - 2p - 2
    N_int_knots = N_basis - p - 1
    # uniformly spaced knots
    knots = jnp.linspace(0, 1, N_int_knots+2)
    # internal knots: points where the cubic polynomials are joined
    int_knots = knots[1:-1]
    # open/clamped splines: repeat endpoints p+1 times => 1st endpoint has 
    # spline coordinate t = 0, whereas last point has coordinate t = 1
    knots_all = jnp.concatenate((jnp.zeros(p+1), int_knots, jnp.ones(p+1)))
    
    # evaluate the basis functions for all the values of the non-dimensional
    # parameter (x/L for Bx, t/T for Bt)
    #   [B0(xi_0) ... Bn(xi_0)]
    #   [ ...   ...   ... ]
    #   [B0(xi_N) ... Bn(xi_N)]
    # i.e. Bij = Bj(xi_i)
    if domain is None:
        L = xi[-1] - xi[0]
    else:
        # if interface nodes are passed, L calculated from the mesh would be
        # L - dx/2 - dx/2 = L - dx
        L = domain[-1] - domain[0]
    xi = jnp.asarray(xi) / L
    B = BSpline.design_matrix(xi, knots_all, p)
    # convert to an actual (NumPy) matrix, then to a JAX array
    #   [B0(xi_0) ... B0(xi_N)]
    #   [ ...   ...   ... ]
    #   [Bn(xi_0) ... Bn(xi_N)]
    B_mat = jnp.asarray(B.toarray().T)
    
    if PLOT:
        plt.figure()
        for n in range(N_basis):
            plt.plot(xi, B_mat[n,:], label=f'$B_{n}$')
        plt.xlabel(r'$\xi$ [-]')
        plt.ylabel(r'$B_i$ [-]')
        plt.xlim(0, 1)
        plt.legend()
        plt.title(f'{N_basis} cubic B-splines')
        plt.show()
    
    return B_mat


#@partial(jax.jit, static_argnames=('h_beta', ))
def cost_function(beta, d, h_beta, Cm, Cbeta, beta_prior):
    sol_beta = h_beta(beta)
    cost_data = _quadratic_form(d-sol_beta, Cm)
    cost_beta = _quadratic_form(beta-beta_prior, Cbeta)
    J = cost_data + cost_beta
    return J


# no need to add it here: _quadratic_form() is used only in a single function
# and that function is already JIT-compiled
#@jax.jit
def _quadratic_form(x, A):
    # make sure that x is (N, ) and not (N, 1) or similar non-sense
    x = jnp.ravel(jnp.asarray(x))
    A = jnp.asarray(A)
    
    if A.ndim == 0:
        # scalar matrix: A = a * I, then A^-1 = (1/a) I
        # x^TAx = x^T(I/a)x = (1/a) x^Tx = (1/a) ||x||^2
        Q = 0.5 * jnp.vdot(x, x) / A
    elif A.ndim == 1:
        # A is a vector of variances (i.e. diagonal matrix)
        # product of a diagonal matrix and a vector is just y_i = B_ii * x_i
        # => x^T * A^-1 * x = sum_i x_i A^-1_ii x_i
        A_inv = 1 / A   # only diagonal elements
        Q = 0.5 * jnp.sum(x**2 * A_inv)
    elif A.ndim == 2:
        # full covariance matrix (not diagonal)
        # x^T A^-1 x = x^T (A^-1 x) = x^T y
        # where y is the solution of the system Ay = x => more efficient and
        # stable than calculating the inverse with jnp.inv(A)
        Q = 0.5 * x.T @ (jnp.linalg.solve(A, x))
    return Q