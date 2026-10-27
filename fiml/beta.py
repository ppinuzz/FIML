"""
Utilities to construct beta function.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


def read_beta_options(input_data):
    # DEFAULT VALUES
    # used in beta = beta_ref + ...
    beta_ref = 1
    # used in beta = beta_min + (1-beta)*sofplus(...), so that beta > beta_min
    beta_min = 1e-2
    
    beta_opts = {'beta_ref': beta_ref,
                 'beta_min': beta_min,
                 # rule of thumb: around 10 RBF per dimension => MAX 20 RBFs
                 'N_RBF': 15,
                 # shape factor K in exp(-K*x^2)
                 'shape_factor': 1
                 }
    # overwrite default values, if given
    for key in beta_opts:
        if key in input_data and input_data[key] is not None:
            beta_opts[key] = input_data[key]
    
    beta_opts['model'] = input_data['beta_type']
    match beta_opts['model']:
        case 'BxBt' | 'BxBt_softplus' | 'BxBt_exp':
            beta_opts['N_basis_x'] = int(input_data['num_basis_functions_x'])
            beta_opts['N_basis_t'] = int(input_data['num_basis_functions_t'])
        case 'RBF_Gauss_x' | 'RBF_Gauss_u':
            beta_opts['N_RBF'] = int(input_data['num_RBF'])
            beta_opts['K'] = float(input_data['shape_factor'])
        case _:
            raise ValueError(f"Unknown beta model '{beta_opts['model']}'")
    return beta_opts


def parametrise_beta(opts, x, t, STEADY):
    # calculate parameters common to several models
    match opts['model']:
        case 'BxBt' | 'BxBt_softplus' | 'BxBt_exp':
            # matrices containing
            #   [B0(t0) ... B0(tN)]
            #   [ ...   ...   ... ]
            #   [Bn(t0) ... Bn(tN)]
            # where t can be either time (for B_t) or space (for B_x)
            print('Calculating basis function matrices...')
            B_x = create_basis_functions_matrix(opts['N_basis_x'], x)
            if not STEADY:
                B_t = create_basis_functions_matrix(opts['N_basis_t'], t)
            else:
                # "identity matrix" with only 1 element
                B_t = jnp.eye(1)
            # same, but evaluated at interface nodes
            # (N nodes => N-1 interfaces => N-1 interface nodes)
            #x_interf = 0.5 * (x[:-1] + x[1:])
            # specify the original domain mesh, so that its length is okay
            #B_x_interf = create_basis_functions_matrix(opts['N_basis_x'], x_interf, domain=x)
            N_params = B_x.shape[0] * B_t.shape[0]
        case 'RBF_Gauss_x' | 'RBF_Gauss_u':
            print('Calculating Gaussian radial basis functions matrix...')
            x_c = jnp.linspace(x[0], x[-1], opts['N_RBF'])
            # j-th column is the vector of values of phi_j(x)
            Phi = jax.vmap(lambda xc: jnp.exp(-opts['K']*(x-xc)**2),
                           in_axes=0, out_axes=1)(x_c)
            N_params = opts['N_RBF']
        case _:
            raise ValueError(f"Unknown beta model '{opts['model']}'")
    print(f'Number of theta parameters to identify: {N_params}')
    
    # wrap each beta function
    match opts['model']:
        case 'BxBt':
            beta = lambda theta: _beta_BxBt(theta, B_x, B_t, opts['beta_ref'])
        case 'BxBt_softplus':
            beta = lambda theta: _beta_BxBt_softplus(theta, B_x, B_t, opts['beta_min'])
        case 'BxBt_exp':
            beta = lambda theta: _beta_BxBt_exp(theta, B_x, B_t, opts['beta_ref'])
        case 'RBF_Gauss_x':
            beta = lambda theta: _beta_RBF_Gauss_x(theta, Phi, t.size, opts['beta_ref'])
        case _:
            raise ValueError(f"Unknown beta model '{opts['model']}'")
    
    return beta, N_params


def _beta_BxBt(theta_r, B_x, B_t, beta_ref):    
    N_basis_x = B_x.shape[0]
    N_basis_t = B_t.shape[0]
    # theta = vector of vectors 
    #   [theta_11, theta_12, ..., theta_1Q, theta_21, ..., theta_MQ]
    #   = [theta_1; theta_2; ..., theta_M]
    # the m-th component contains Q coefficients for each one of the Q time-functions
    # in 1 + sum_mq theta_mq Bm(x) Bq(t)
    # transform it to a matrix [theta_1, theta_2, ..., theta_M] (M x Q)
    Theta_r = jnp.reshape(theta_r, (N_basis_x, N_basis_t))
    # correction for the entire space-time mesh (N_x x N_t)
    beta_r = beta_ref + B_x.T @ Theta_r @ B_t
    return beta_r

def _beta_BxBt_softplus(theta_r, B_x, B_t, beta_min):
    z = _beta_BxBt(theta_r, B_x, B_t, beta_ref=0)
    beta_r = beta_min + (1 - beta_min)*jax.nn.softplus(z)/jnp.log(2)
    return beta_r

def _beta_BxBt_exp(theta_r, B_x, B_t, beta_ref):
    # beta = exp(z), with z = sum_mq Bm(x) theta_mq Bq(t)
    z = _beta_BxBt(theta_r, B_x, B_t, beta_ref=0)
    beta_r = jnp.exp(z)
    return beta_r

def _beta_RBF_Gauss_x(theta_r, Phi, N_t, beta_ref):
    beta_r = beta_ref + Phi @ theta_r
    # to keep it consistent with the unsteady case, beta must be (Nx, Nt)
    # => this RBF is steady, manually add the "time dimension"
    beta_r = jnp.broadcast_to(beta_r[:,jnp.newaxis], (Phi.shape[0], N_t))
    return beta_r


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
        domain = [xi[0], xi[-1]]
    # if interface nodes are passed, L calculated from the mesh would be
    # L - dx/2 - dx/2 = L - dx
    L = domain[-1] - domain[0]
    # if original domain is [3, 7], but the interface mesh [3.5, ..., 6.5] is
    # passed to the function, you cannot shift xi by 3.5, as the 1st point 
    # is not at xi = 0, but at xi = (3.5-3)/4
    xi = (jnp.asarray(xi) - domain[0]) / L
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
