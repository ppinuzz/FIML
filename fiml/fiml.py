"""
Core FIML utilities.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

from os import process_cpu_count
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.random as jrn
#from functools import partial
import fiml.parallel as parallel
import fiml.auxiliaries as aux


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

#@jax.jit
def beta(theta_r, B_x, B_t, NO_NEG=False, BETA_MIN=1e-2):
    beta_r = _beta_basis_functions(theta_r, B_x, B_t)
    if NO_NEG:
        # TODO: find a way to avoid having nu < 0 (i.e. not having beta < 0)
        #beta_r = jnp.maximum(beta_r, BETA_MIN)
        beta_r = BETA_MIN + (1.0 - BETA_MIN)*jax.nn.softplus(beta_r-1)/jnp.log(2)
    
    return beta_r


def _beta_basis_functions(theta_r, B_x, B_t):
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
    beta_r = 1 + B_x.T @ Theta_r @ B_t
    
    return beta_r


def _beta_basis_functions_exp(theta_r, B_x, B_t):
    # only matrix product Bx^T * Theta * Bt
    beta_r = _beta_basis_functions(theta_r, B_x, B_t) - 1
    beta_r = jnp.exp(beta_r)
    return beta_r


def prior_sampling(theta_prior, var_prior, N_samples, key):
    theta_prior = jnp.asarray(theta_prior)
    # shape (1,) has ndim = 1, but it's a scalar => remove dimensions equal to 1
    var_prior = jnp.squeeze(var_prior)
    N_theta = theta_prior.size
    
    if var_prior.ndim == 0:
        # extract from normal N(0,1), move to Gaussian N(mu,sigma^2)
        theta_prior_samples = (theta_prior[:,jnp.newaxis]
                                   + jnp.sqrt(var_prior) * jrn.normal(
                                       key, shape=(N_theta, N_samples))
                            )
    elif var_prior.ndim == 1:
        theta_prior_samples = (theta_prior[:,jnp.newaxis]
                               + jnp.sqrt(var_prior)[:,jnp.newaxis] * jrn.normal(
                                   key, shape=(N_theta, N_samples))
                               )
    elif var_prior.ndim == 2:
        # since the output is a theta vector, each vector is a single element
        # of the output => returned as a matrix (?) where each row is a different
        # sample => transpose it
        theta_prior_samples = jrn.multivariate_normal(key,
                                                      mean=theta_prior,
                                                      cov=var_prior,
                                                      shape=(N_samples,)).T
    return theta_prior_samples


def posterior_sampling(theta_MAP, C_beta_MAP, N_samples, key):
    R_chol = jnp.linalg.cholesky(C_beta_MAP)
    # Eq. (10) and Eq. (11) in [Parish2016]
    s_random = jrn.normal(key, shape=(theta_MAP.size, N_samples))
    theta_post_samples = jax.vmap(lambda s: theta_MAP + R_chol @ s, 
                                  in_axes=1, out_axes=1)(s_random)
    return theta_post_samples


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


def launch_sampling(theta_samples, run_solver_case, solver_fun, N_x, N_t, N_samples,  solver_args, sample_dir=None, RUN_PARALLEL=False, N_WORKERS=1):
    solution = jnp.zeros(shape=(N_t, N_x, N_samples))
    # 2nd dimension 1 is not really needed, but that way they have the same
    # calling scheme (time, space, sample)
    max_residual = jnp.zeros(shape=(N_t, 1, N_samples))
    # the array will contain numbers 0/1, but they will be treated as booleans
    # (because int(0) == False and int(1) == True)
    success = jnp.zeros(shape=(N_samples, ), dtype=bool)
    if RUN_PARALLEL:
        print('-------------- PARALLEL RUN --------------')
        print(f'Starting workers pool with (at most) {N_WORKERS} workers...')
        print(f'Max number of logical CPU cores this program can call: {process_cpu_count()}')
        # 2nd argument is the object looped through => 'for' loops the ROWS
        # of the matrix, but here it should loop the columns of theta_prior_samples
        # (each column is a different sample) => loop through the transpose
        solution, max_residual, success = parallel.run_parallel(N_WORKERS, 
                                        theta_samples.T, run_solver_case, 
                                        solution, max_residual, success,
                                        solver_fun, *solver_args, sample_dir)
    else:
        print('-------------- SEQUENTIAL RUN --------------')
        for r in range(theta_samples.shape[1]):
            print(f'Running sample {r}')
            theta_r = theta_samples[:,r]
            _, sol, max_res, status = run_solver_case(
                        r, theta_r, solver_fun, *solver_args, sample_dir)
            solution = solution.at[:,:,r].set(sol)
            max_residual = max_residual.at[:,0,r].set(max_res)
            success = success.at[r].set(status)

    run_type = 'Parallel' if RUN_PARALLEL else 'Sequential'
    print(f'{run_type} execution finished, data have been written to \n\t{sample_dir.resolve()}')
    return solution, max_residual, success