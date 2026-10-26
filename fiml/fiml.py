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
from prettytable import PrettyTable


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


def _beta_basis_functions(theta_r, B_x, B_t, beta_ref=1):
    
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


def posterior_sampling(theta_MAP, C_theta_MAP, N_samples, key):
    R_chol = jnp.linalg.cholesky(C_theta_MAP)
    # Eq. (10) and Eq. (11) in [Parish2016]
    s_random = jrn.normal(key, shape=(theta_MAP.size, N_samples))
    theta_post_samples = jax.vmap(lambda s: theta_MAP + R_chol @ s, 
                                  in_axes=1, out_axes=1)(s_random)
    return theta_post_samples


def remove_failed_runs(run_data, converged):
    N_runs = converged.size
    N_failed = N_runs - sum(converged)
    # flag failed runs and remove their data to avoid skewing the distribution
    if (~converged).any():
        aux.warning(f'{N_failed}/{N_runs} simulation failed\n'
                'Removing failed simulation data...')
        clean_data = []
        for data in run_data:
            # ellipsis (...) is the same as ':' for all previous dimensions
            data = data[..., converged]
            clean_data.append(data)
        aux.warning(f'Number of samples modified from {N_runs} (user input) to {sum(converged)} (non-failed simulations)...')
    else:
        clean_data = run_data
    # unpack the list: specify they're part of the same 1-element return tuple
    # https://stackoverflow.com/a/74318655/17220538
    return (*clean_data, )


def handle_failed_runs(run_data, success, strategy='remove', MAX_FAIL_RATE=0.01):
    N_failed = int(jnp.count_nonzero(~success))
    N_total = success.size
    fail_rate = N_failed / N_total
    if fail_rate > MAX_FAIL_RATE:
        aux.warning('Failure rate exceeds maximum limit: '
                    f'{N_failed}/{N_total} = {fail_rate} > {MAX_FAIL_RATE}')
    match strategy.lower():
        case 'remove':
            clean_data = remove_failed_runs(run_data, success)
        case _:
            raise ValueError(f"Unknown failure strategy '{strategy}'")
    return clean_data


def compare_baseline_MAP_solutions(sol_MAP, theta_baseline, solver_wrap, 
                                   sol_exp, x, x_exp, t=None, t_exp=None,
                                   PLOT=True, fig_name=None):
    # baseline solution (beta = 1 => theta = 0)
    _, sol_baseline, _, _ = solver_wrap(theta_baseline)
    # sample both solutions at probe locations
    sol_MAP_probes = aux.sample_solution(sol_MAP, x_mesh=x, t_mesh=t, 
                                         x_probes=x_exp, t_probes=t_exp)
    sol_baseline_probes = aux.sample_solution(sol_baseline, x_mesh=x, t_mesh=t, 
                                              x_probes=x_exp, t_probes=t_exp)
    # errors w.r.t. experimental data (actual truth)
    err_MAP = sol_MAP_probes - sol_exp
    err_baseline = sol_baseline_probes - sol_exp
    
    # mean absolute error > 0
    MAE_MAP = jnp.mean(jnp.abs(err_MAP))
    MAE_baseline = jnp.mean(jnp.abs(err_baseline))
    # root mean square error > 0
    RMSE_MAP = jnp.sqrt(jnp.mean(err_MAP**2))
    RMSE_baseline = jnp.sqrt(jnp.mean(err_baseline**2))
    
    err_report = PrettyTable()
    err_report.title = 'Error report: MAP and baseline VS experiments'
    err_report.field_names = ['Metric', 'Baseline', 'MAP']
    err_report.add_row(['Avg. MAE [EU]', f'{MAE_baseline:.3e}', f'{MAE_MAP:.3e}'])
    err_report.add_row(['Avg. RMSE [EU]', f'{RMSE_baseline:.3e}', f'{RMSE_MAP:.3e}'])
    aux.check_msg(err_report.get_string())
    
    # MAE and RMSE @ each x_probe => average over time
    MAE_MAP_probe = jnp.mean(jnp.abs(err_MAP), axis=0)
    MAE_baseline_probe = jnp.mean(jnp.abs(err_baseline), axis=0)
    RMSE_MAP_probe = jnp.sqrt(jnp.mean(err_MAP**2, axis=0))
    RMSE_baseline_probe = jnp.sqrt(jnp.mean(err_baseline**2, axis=0))

    # delta > 0: improvement w.r.t. baseline model
    delta_MAE = MAE_baseline_probe - MAE_MAP_probe
    delta_RMSE = RMSE_baseline_probe - RMSE_MAP_probe
    # fraction of probes where MAP outperforms baseline
    N_delta_MAE = jnp.count_nonzero(delta_MAE > 0) / delta_MAE.size
    N_delta_RMSE = jnp.count_nonzero(delta_RMSE > 0) / delta_RMSE.size
    # signed mean improvements: negative values mean deterioration
    avg_delta_MAE = jnp.mean(delta_MAE)
    avg_delta_RMSE = jnp.mean(delta_RMSE)
    delta_report = PrettyTable()
    delta_report.title = 'MAP improvement over baseline'
    delta_report.field_names = ['Metric', '% of probes improved', 'Mean improvement [EU]']
    delta_report.add_row(['MAE', f'{100*N_delta_MAE:.2f}', f'{avg_delta_MAE:.3e}'])
    delta_report.add_row(['RMSE', f'{100*N_delta_RMSE:.2f}', f'{avg_delta_RMSE:.3e}'])
    aux.check_msg(delta_report.get_string())

    # fraction of all observed space-time points with lower absolute error
    pointwise_improvement = jnp.abs(err_baseline) - jnp.abs(err_MAP)
    fraction_improved = jnp.mean(pointwise_improvement > 0)
    aux.check_msg(f'MAP has lower absolute error at '
                  f'{100*fraction_improved:.2f}% of experimental data points')
    
    if PLOT:
        fig = plt.imshow(jnp.abs(err_baseline)-jnp.abs(err_MAP), cmap='hot')
        plt.colorbar()
        plt.show(block=False)
        if fig_name is not None:
            # plt.imshow() doesn't return a Figure, but an Axes object
            aux.save_fig(fig.figure, fig_name.stem, fig_name.parent)


#@partial(jax.jit, static_argnames=('h_beta', ))
def cost_function(theta, d, h_theta, Cm, C_theta, theta_prior):
    sol_theta, success = h_theta(theta)
    cost_data = _quadratic_form(d-sol_theta, Cm)
    cost_theta = _quadratic_form(theta-theta_prior, C_theta)
    J = cost_data + cost_theta
    return J, success


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


def parametrise_beta(opts, x, t, STEADY):
    
    if opts['model'] in {'BxBt', 'BxBt_softplus', 'BxBt_exp'}:
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
        print(f'Number of theta parameters to find: {N_params}')
    
    match opts['model']:
        case 'BxBt':
            beta = lambda theta: _beta_BxBt(theta, B_x, B_t, opts['beta_ref'])
        case 'BxBt_softplus':
            beta = lambda theta: _beta_BxBt_softplus(theta, B_x, B_t, opts['beta_min'])
        case 'BxBt_exp':
            beta = lambda theta: _beta_BxBt_exp(theta, B_x, B_t, opts['beta_ref'])
        case _:
            raise ValueError(f"Unknown beta model '{opts['model']}'")
    
    return beta, N_params


def _beta_BxBt(theta_r, B_x, B_t, beta_ref):
    beta_r = _beta_basis_functions(theta_r, B_x, B_t, beta_ref=beta_ref)
    return beta_r

def _beta_BxBt_softplus(theta_r, B_x, B_t, beta_min):
    z = _beta_basis_functions(theta_r, B_x, B_t, beta_ref=0)
    beta_r = beta_min + (1 - beta_min)*jax.nn.softplus(z)/jnp.log(2)
    return beta_r

def _beta_BxBt_exp(theta_r, B_x, B_t, beta_ref):
    # beta = exp(z), with z = sum_mq Bm(x) theta_mq Bq(t)
    z = _beta_basis_functions(theta_r, B_x, B_t, beta_ref=0)
    beta_r = jnp.exp(z)
    return beta_r


def read_beta_options(input_data):
    # DEFAULT VALUES
    # used in beta = beta_ref + ...
    beta_ref = 1
    # used in beta = beta_min + (1-beta)*sofplus(...), so that beta > beta_min
    beta_min = 1e-2
    
    beta_opts = {'beta_ref': beta_ref,
                 'beta_min': beta_min
                 }
    # overwrite default values, if given
    for key in beta_opts:
        if key in input_data:
            beta_opts[key] = input_data[key]
    
    beta_opts['model'] = input_data['beta_type']
    match beta_opts['model']:
        case 'BxBt' | 'BxBt_softplus' | 'BxBt_exp':
            beta_opts['N_basis_x'] = int(input_data['num_basis_functions_x'])
            beta_opts['N_basis_t'] = int(input_data['num_basis_functions_t'])
        case _:
            raise ValueError(f"Unknown beta model '{beta_opts['model']}'")
    return beta_opts


def launch_sampling(theta_samples, run_solver_case, solver_fun, N_x, N_t, N_samples, solver_args, sample_dir=None, RUN_PARALLEL=False, N_WORKERS=1):
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
    print(f'{run_type} execution finished')
    return solution, max_residual, success