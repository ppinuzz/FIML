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