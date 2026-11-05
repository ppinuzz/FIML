"""
Core FIML utilities.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

from os import process_cpu_count
import matplotlib.pyplot as plt
import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.random as jrn
#from functools import partial
import fiml.parallel as parallel
import fiml.auxiliaries as aux
from prettytable import PrettyTable
import scipy.optimize as sciopt
import optax
import numpy as np


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


def Gauss_Newton_hessian(h_fun, theta_MAP, Cm, Ctheta):
    # h_theta returns the solution & the success flag => specify that
    # it returns a pair and only the 1st item is the actual output
    # G_ij = d(h(theta)_i)/dtheta_j
    G_MAP = jax.jacfwd(h_fun)(theta_MAP)    # (N_data, N_theta)
    
    # G^T * Cm^-1 * G = (N_data, N_theta)^T * Cm * (N_data, N_theta)
    # therefore Cm = (N_data, N_data)
    Cm_inv = invert_covariance_matrix(Cm, at_least_2d=True, N=G_MAP.shape[0])
    H_1 = G_MAP.T @ Cm_inv @ G_MAP
    H_2 = invert_covariance_matrix(Ctheta, at_least_2d=True, N=theta_MAP.shape[0])
    H_GN = H_1 + H_2
    return H_GN


def invert_covariance_matrix(C, at_least_2d=False, N=None):
    if at_least_2d and N is None:
        raise ValueError('If C is a scalar and a 2D covariance matrix is required, '
                         'it is mandatory to provide the dimension N of the matrix')
    C = jnp.asarray(C)
    if C.ndim == 0:
        C_inv = 1 / C
        if at_least_2d:
            C_inv = C_inv * jnp.eye(N)
    elif C.ndim == 1:
        C_inv = jnp.diag(1/C)
    else:
        # solve the linear system C*C^-1 = I to find C^-1 (Jordan's method?)
        C_inv = jnp.linalg.solve(C, jnp.eye(C.shape[0]))
    return C_inv


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


def minimise_cost_function(cost_and_jacobian, x0, method='amsgrad',
                           learn_rate=1e-2, max_iter=500, tol=1e-5,
                           max_backtrack=8, print_every=20):
    
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    match method.lower():
        case 'adam':
            optimiser = optax.adam(learn_rate)
        case 'amsgrad':
            optimiser = optax.amsgrad(learn_rate)
        case _:
            raise ValueError(f"Unknown adaptive optimiser '{method}'")

    # 3 checks on the solution:
    #   - Burgers solver converged
    #   - objective function is finite
    #   - gradient of objective function is finite
    def _is_valid_sol(cost, grad, forward_ok):
        okay_sol = forward_ok & jnp.isfinite(cost) & jnp.all(jnp.isfinite(grad))
        return okay_sol

    @jax.jit
    def run_optimiser(theta_0):
        # initialise optimiser internal state with estimates for the 1st and 
        # 2nd moments
        int_state = optimiser.init(theta_0)
        cost_0, grad_0, success_0 = cost_and_jacobian(theta_0)
        valid_sol_0 = _is_valid_sol(cost_0, grad_0, success_0)
        history = {'cost': jnp.full((max_iter+1,), jnp.nan),
                   'grad_norm': jnp.full((max_iter+1,), jnp.nan),
                   'theta': jnp.full((max_iter+1, x0.size), jnp.nan),
                   'URF': jnp.full((max_iter+1,), jnp.nan),
                   'N_fun_eval': jnp.full((max_iter+1,), jnp.nan)
                   }
        state_0 = (theta_0, int_state, cost_0, grad_0, 0, 1, valid_sol_0, False, history)

        history['cost'] = history['cost'].at[0].set(cost_0)
        history['grad_norm'] = history['grad_norm'].at[0].set(jnp.linalg.norm(grad_0))
        history['theta'] = history['theta'].at[0].set(theta_0)
        # no under-relaxation at the beginning
        history['URF'] = history['URF'].at[0].set(1)
        history['N_fun_eval'] = history['N_fun_eval'].at[0].set(1)
        
        # stopping condition: ||grad(J)|| > tol 
        # (exact solution: grad(J) = 0)
        def continue_optimisation(state):
            _, _, _, grad, N_it, _, valid, _, _ = state
            keep_going = valid & (N_it < max_iter) & (jnp.linalg.norm(grad) > tol)
            return keep_going

        def optimisation_step(state):
            theta, opt_state, cost, grad, N_it, N_fun_eval, valid, failed, history = state
            updates, opt_state_trial = optimiser.update(grad, opt_state, theta)
            # N_attempts, accepted, theta, J, grad(J)
            trial_state = (0, False, theta, cost, grad)

            def continue_backtracking(trial):
                N_attempts, accepted, _, _, _ = trial
                keep_going = (~accepted) & (N_attempts <= max_backtrack)
                return keep_going

            def backtracking_step(trial):
                N_attempts, _, theta_valid, cost_valid, grad_valid = trial
                # halve URF each time: starts with 0.5^0 = 1 (full update)
                # => if doesn't work, N_attempts is increased to 1 => use 0.5^1 
                URF = 0.5**N_attempts
                # theta_k+1 = theta_k + URF*delta_k
                theta_trial = optax.apply_updates(theta, URF * updates)
                cost_trial, grad_trial, success_trial = cost_and_jacobian(theta_trial)
                valid_sol = _is_valid_sol(cost_trial, grad_trial, success_trial)

                # where the updated solution is valid, replace the old solution;
                # keep the old solution otherwise
                new_state = (N_attempts+1, valid_sol, 
                             jnp.where(valid_sol, theta_trial, theta_valid),
                             jnp.where(valid_sol, cost_trial, cost_valid),
                             jnp.where(valid_sol, grad_trial, grad_valid)
                             )
                return new_state

            # state = trial_state
            # while continue_backtracking(state):
            #   state = backtracking_step()
            N_attempts, okay_sol, theta_new, cost_new, grad_new = (
                                    jax.lax.while_loop(continue_backtracking,
                                                       backtracking_step, 
                                                       trial_state)
                                    )
            # update optimiser internal state
            # (jax.tree.map rather than jax.map because Optax's internal state
            # is a PyTree object)
            opt_state_new = jax.tree.map(lambda old, new: jnp.where(okay_sol, new, old),
                                         opt_state, opt_state_trial)
            N_it = N_it + okay_sol.astype(jnp.int32)
            N_fun_eval = N_fun_eval + N_attempts
            
            def update_history(history):
                return {
                    'cost': history['cost'].at[N_it].set(cost_new),
                    'grad_norm': history['grad_norm'].at[N_it].set(jnp.linalg.norm(grad_new)),
                    'theta': history['theta'].at[N_it].set(theta_new),
                    'URF': history['URF'].at[N_it].set(0.5**(N_attempts-1)),
                    'N_fun_eval': history['N_fun_eval'].at[N_it].set(N_fun_eval)
                }
            history = jax.lax.cond(okay_sol, update_history, lambda h: h, history)

            if print_every:
                # always print at 1st step
                jax.lax.cond(okay_sol & ((N_it == 1) | (N_it % print_every == 0)),
                             # to sue method.upper(), you need an f-string, but
                             # jex.debug.print() format the string at runtime
                             # (because n, k, etc. change during execution)
                             # => surround 'k' with double {{...}}
                             lambda _: jax.debug.print(f'{method.upper()} iteration {{k:5d}}: '
                                                       'J = {cost:.8e}, '
                                                       '||grad(J)|| = {norm:.5e}, '
                                                       'backtracking steps = {n}',
                                                       k=N_it, cost=cost_new,
                                                       norm=jnp.linalg.norm(grad_new), 
                                                       # 1 attempt => URF = 1 => no backtracking
                                                       n=N_attempts-1),
                             lambda _: None, operand=None)

            return (theta_new, opt_state_new, cost_new, grad_new, N_it,
                    N_fun_eval, valid & okay_sol, failed | ~okay_sol, history)

        theta, _, cost, grad, N_it, N_fun_eval, valid, failed, history = (
                                    jax.lax.while_loop(continue_optimisation,
                                                       optimisation_step, 
                                                       state_0)
                                    )
        return theta, cost, grad, N_it, N_fun_eval, valid, failed, valid_sol_0, history

    opt_sol = run_optimiser(x0)
    theta, cost, grad, N_it, N_fun_eval, valid, failed, valid_sol_0, history = opt_sol
    # remove unused entries (array has been preallocated to the max number of
    # iterations)
    history = {key: values[0:N_it+1] for key, values in history.items()}

    # repack output into a SciPy's OptimizeResult object, to make the code
    # compatible with SciPy's BFSG optimiser
    converged = valid & (jnp.linalg.norm(grad) <= tol)
    if converged:
        status = 0
        message = 'Gradient norm norm below tol'
    elif not valid_sol_0:
        status = 3
        message = 'Invalid initial objective, gradient, or forward solve'
    elif failed:
        status = 2
        message = 'No valid Adam/AMSGrad step after backtracking'
    else:
        status = 1
        message = 'Maximum number of iterations reached'

    result = sciopt.OptimizeResult(x=np.asarray(theta), fun=float(cost),
                                   jac=np.asarray(grad), nit=int(N_it),
                                   nfev=int(N_fun_eval), njev=int(N_fun_eval),
                                   success=bool(converged), status=status, 
                                   message=message, history=history)
    return result
