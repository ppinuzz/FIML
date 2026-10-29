"""
Main functions of FIML methodology.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import perf_counter
import yaml

import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import scipy.optimize as sciopt
import h5py

import fiml.fiml as fiml
import fiml.auxiliaries as aux
import fiml.post as post
import fiml.beta as betamod


def run_solver_case(run_id, theta_r, solver_fun, beta_fun, nu, x_mesh, t_mesh, 
                    u_IC, u0, uL, STEADY, out_dir, TOL=1e-5, PRINT_ERR_MSG=True):
    
    # apply ICs and BCs + solve corrected Burgers' eq.
    solution = jnp.zeros(shape=(t_mesh.size, x_mesh.size))
    if not STEADY:
        solution = solution.at[0,:].set(u_IC)
    solution = solution.at[:,0].set(u0)
    solution = solution.at[:,-1].set(uL)
    # getattr(beta_fun, 'depends_on_u') will crash if the attribute is not
    # present => supply False as default value
    if getattr(beta_fun, 'depends_on_u', False):
        # beta = beta(..., u) => pass the callable beta_fun() to the solver
        # => this will trigger the solver into using a different internal
        # function, since the Jacobian is no more tridiagonal
        solution, max_res = solver_fun(solution, x_mesh, t_mesh, nu, 
                                       beta=beta_fun, theta=theta_r,
                                       WRITE_LOG=False, tol=TOL, STEADY=STEADY)
    else:
        # beta = beta(x,t) can be calculated beforehand and doesn't depend on
        # the solution => can choose a faster solver with a tridiag. Jacobian
        beta_r = beta_fun(theta_r)
        solution, max_res = solver_fun(solution, x_mesh, t_mesh, nu, beta=beta_r,
                                       WRITE_LOG=False, tol=TOL, STEADY=STEADY)
    
    # check if solver didn't converge or produced NaN's/Inf's
    not_converged = (max_res >= TOL)
    nan_inf_values = jnp.any(~jnp.isfinite(solution), axis=1) | ~jnp.isfinite(max_res)
    # jax.lax.cond() replaces if (...) when the object is a JAX object in a JIT
    # compiled function
    jax.lax.cond(PRINT_ERR_MSG & jnp.any(not_converged), 
                 # jax.debug.print() replaces print() in JIT-compiled functions
                 lambda _: jax.debug.print(
                             '{n_fail}/{n_tot} time instants did not converge for run {run_id}',
                             n_fail=jnp.count_nonzero(not_converged),
                             n_tot=not_converged.size,
                             run_id=run_id),
                 lambda _: None,
                 operand=PRINT_ERR_MSG
                 )
        
    jax.lax.cond(PRINT_ERR_MSG & jnp.any(nan_inf_values), 
                 lambda _: jax.debug.print(
                             '{n_nan}/{n_tot} time instants returned a NaN/Inf '
                             'solution and/or residual for run {run_id}',
                             n_nan=jnp.count_nonzero(nan_inf_values),
                             n_tot=nan_inf_values.size,
                             run_id=run_id),
                 lambda _: None,
                 operand=PRINT_ERR_MSG
                 )
    success = ~jnp.any(not_converged | nan_inf_values)
    
    return run_id, solution, max_res, success


def analysis(exp_data, mesh, BC, nu, out_dir, beta_opts, sigma2_prior, 
             N_sampling_prior, N_sampling_post, SEED, sigma2_obs, N_WORKERS, RUN_PARALLEL,
             solver_fun, CHECK=False, VIDEO=False, beta_exact=None, BATCH=False,
             IDX_TIME_PLOT=-1, SAVE_EXP=True, PLOT=True, STEADY=True, hessian='full',
             input_file=None):
    
    if BATCH:
        aux.warning('BATCH mode selected, turning off all plot flags...')
        CHECK = False
        VIDEO = False
        PLOT = False
    
    # OUTPUT DATA FILE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = out_dir / 'output.h5'
    out_file = h5py.File(out_filename, 'w')
    # common compression options
    # shuffle: allows blocks to be moved around, possible memory gain
    # gzip: most portable and fairly fast
    # compression_opts: for GZIP, it's the compressione level (9: maximum)
    comprex_opts = {'shuffle': True, 'compression': 'gzip', 'compression_opts': 9}
    
    # MESH DATA
    x_mesh = jnp.asarray(mesh['x'])
    t_mesh = jnp.asarray(mesh['t'])
    N_x = x_mesh.size
    N_t = t_mesh.size
    L = x_mesh[-1] - x_mesh[0]
    t0 = t_mesh[0]
    if N_t == 1:
        aux.warning('Steady problem: initial conditions, time length and dt will be ignored')
        T = None
        dt = None
        STEADY = True
    else:
        T = t_mesh[-1] - t_mesh[0]
        dt = t_mesh[1] - t_mesh[0]
        STEADY = False
        if not jnp.all(jnp.isclose(jnp.diff(t_mesh), dt)):
            raise ValueError('Only uniform time grids are supported')
    # HACK: assuming uniform spacing in time and space
    dx = x_mesh[1] - x_mesh[0]
    if not jnp.all(jnp.isclose(jnp.diff(x_mesh), dx)):
        raise ValueError('Only uniform space grids are supported')
    
    hdf5_group = out_file.create_group('mesh')
    tmp_set = hdf5_group.create_dataset(name='x', data=x_mesh, **comprex_opts)
    tmp_set.attrs['description'] = 'Spatial mesh'
    tmp_set = hdf5_group.create_dataset(name='t', data=t_mesh, **comprex_opts)
    tmp_set.attrs['description'] = 'Time mesh'
    # write to disk without closing the file
    out_file.flush()
    
    # NUMERICS
    if STEADY:
        if 'N_basis_t' in beta_opts and beta_opts['N_basis_t'] is not None:
            aux.warning("Steady problems should not use time basis-function matrix B_t, "
                        f"but {beta_opts['N_basis_t']} time basis functions are given, "
                        "skipping them...")
        beta_opts['N_basis_t'] = None
    
    
    # BOUNDARY CONDITIONS
    u0 = BC['u0']
    uL = BC['uL']
    u_IC_fun = BC['u_IC']                   # function u0(x)
    if STEADY:
        u_IC = None
    else:
        # numerical discretisation of u0
        u_IC = jnp.asarray(u_IC_fun(x_mesh))
    
    # EXPERIMENTAL DATA
    x_exp = jnp.asarray(exp_data['x'])
    t_exp = jnp.asarray(exp_data['t'])
    sol_exp = jnp.asarray(exp_data['solution'])
    
    if SAVE_EXP:
        # save experimental data to file using the same format as solution data
        # (for reproducibility)
        exp_out_file = h5py.File(out_dir/'experiments.h5', 'w')
        tmp_set = exp_out_file.create_dataset(name='solution', data=sol_exp, **comprex_opts)
        tmp_set.attrs['description'] = ('The i-th row contains all the experimental data at the i-th time instant, '
                                        'each column is a different probe')
        tmp_set = exp_out_file.create_dataset(name='x', data=x_exp, **comprex_opts)
        tmp_set.attrs['description'] = 'Probes coordinates'
        tmp_set = exp_out_file.create_dataset(name='t', data=t_exp, **comprex_opts)
        tmp_set.attrs['description'] = 'Time instants acquired in the experiment'
        # write to disk without closing the file
        print(f'Writing experimental data to \n\t{Path(exp_out_file.filename).resolve()}')
        exp_out_file.close()
    
    if not STEADY:
        # check that experimental data and simulation data start and end at the same
        # time instant (or that, at least, the simulation is longer than the experiment)
        # discretisation with step dt => any number within -dt/2 and +dt/2 is associated
        # to the instant 0
        if t_exp[0] < t0-0.5*dt or t_exp[0] > t0+0.5*dt:
            raise ValueError('Experimental data and simulation must start at the same time instant '
                             '(within a dt/2)\n'
                             f'Experimental data t0  : {t_exp[0]} s\n'
                             f'Simulation t0         : {t0} s\n')
        if T < t_exp[-1]:
            aux.warning('Numerical simulation is shorter than the experimental data: \n'
                        f'T numerical: {T} s\n'
                        f'T experiment: {t_exp[-1]} s\n'
                        f'Limiting experimental data to numerical simulation end time T = {T} s')
            idx_valid = (t_exp <= T)
            t_exp = t_exp[idx_valid]
            sol_exp = sol_exp[idx_valid, :]
    
    
    # copy content of the input file to output (for reproducibility's sake)
    
    if input_file is not None:
        print(f'Saving input data to \n\t{out_filename.resolve()}...')
        hdf5_group = out_file.create_group('input_data')
        with open(input_file, 'r') as file:
            input_data = yaml.safe_load(file)
        aux.save_dict_to_hdf5(input_data, hdf5_group)
        out_file.flush()
    else:
        # "float" empty type
        out_file.create_dataset(name='input_data', data=h5py.Empty('f'))
    
    
    #%% PARAMETRISE CORRECTION FUNCTION BETA
    
    beta_fun, N_theta = betamod.parametrise_beta(beta_opts, x_mesh, t_mesh, STEADY)
    
    
    #%% DRAW FROM PRIOR DISTRIBUTION
    
    # check that covariance matrix is finite, symmetryc, positive semi-definite
    aux.check_covariance_matrix(sigma2_prior)
    # beta = 1 + sum_mq theta_mq Bm(x) Bq(t)
    # beta_prior = 1 (no correction) means theta_prior = 0
    theta_prior = np.zeros(shape=(N_theta, ))
    key_random = aux.generate_key(SEED)
    # each column of the matrix is one draw of the theta vector, which contains
    # the N_theta parameters of the beta(x,t) function
    print(f'Drawing {N_sampling_prior} samples of the {N_theta}-components theta vector...')
    theta_prior_samples = fiml.prior_sampling(theta_prior, sigma2_prior, 
                                              N_sampling_prior, key_random)
    

    #%% SOLVE CORRECTED BURGERS' EQUATION
    
    solution_corrected, max_res_corrected, success_corrected = fiml.launch_sampling(theta_prior_samples, run_solver_case, solver_fun, N_x, N_t, 
                    N_sampling_prior, (beta_fun, nu, x_mesh, t_mesh, u_IC, u0, uL, STEADY), RUN_PARALLEL=RUN_PARALLEL, N_WORKERS=N_WORKERS)
    
    beta_corrected = jax.vmap(lambda theta, u: beta_fun(theta, u=u),
                              # theta is a row vector of columns => pick one column
                              # of theta at a time => index 1
                              # u is ordered as a (Nx, Nt, N_samples) matrix =>
                              # calculate beta for each sample solution => index 2
                              in_axes=(1, 2),
                              # store results in the 3rd dimension, "one per layer"
                              out_axes=2
                              )(theta_prior_samples, solution_corrected)
    aux.save_sampling(out_file, 'prior', comprex_opts, theta_prior_samples, 
                      solution_corrected, max_res_corrected, success_corrected,
                      beta_corrected, jax.random.key_data(key_random))
    
    solution_corrected, max_res_corrected = fiml.handle_failed_runs(
                                                (solution_corrected, max_res_corrected), 
                                                success_corrected, 
                                                strategy='remove')
    N_sampling_prior = solution_corrected.shape[-1]
    if N_sampling_prior == 0:
        raise RuntimeError('All simulations failed during prior sampling, no data to '
                           'evaluate prior distribution')
        
    
    #%% CHECK: DO THE OBSERVED DATA LIE WITHIN +/- 2*SIGMA?
    
    post.check_experiments_in_sigma_band(solution_corrected, sol_exp, x_mesh, 
                                         t_mesh, x_exp, t_exp, N_sigma=2, 
                                         t_plot=t_exp[IDX_TIME_PLOT], 
                                         fig_file=out_dir/'meanSigmaExpPrior', PLOT=PLOT)
    
    
    #%% FIND MAXIMUM-A-POSTERIORI BETA/THETA
    
    def h_theta_sampled(theta, beta_fun, nu, x_mesh, t_mesh, u_IC, u0, uL, STEADY,
                        x_probes, t_probes):
        # run_id, solution, max_res
        # k-th row of solution: [t^k, u_1^k, ..., u_Nx^k] = [t^k, u(t^k)]
        _, sol, _, success = run_solver_case(None, theta, solver_fun, beta_fun, nu, 
                                             x_mesh, t_mesh, u_IC, u0, uL, STEADY,
                                             out_dir=None, PRINT_ERR_MSG=False)
        # sample numerical solution to have it at the same probe locations
        # [t^k, u_1^k, ...m u_Np^k]     Np = N. of probes
        sol_probes = aux.sample_solution(sol, x_mesh=x_mesh, t_mesh=t_mesh, 
                                         x_probes=x_probes, t_probes=t_probes)
        # create a column vector of column vectors to gather the entire solution
        # as [u(x,t^0), ..., u(x,t^Nt)] => flatten array in column-major order 
        # (Fortran-style) 
        sol_vet = jnp.ravel(sol_probes.T)
        return sol_vet, success
    
    h_theta = lambda theta: h_theta_sampled(theta, beta_fun, nu, x_mesh, t_mesh, 
                                            u_IC, u0, uL, STEADY, x_exp, t_exp)
    # check that covariance matrix is finite, symmetryc, positive semi-definite
    aux.check_covariance_matrix(sigma2_obs)

    def cost_and_status(theta):
        # pass data as a column vector of column vectors
        cost, success = fiml.cost_function(theta, d=jnp.ravel(sol_exp.T), h_theta=h_theta, 
                                  Cm=sigma2_obs, C_theta=sigma2_prior, theta_prior=theta_prior)
        return cost, success

    @jax.jit
    def J_cost_fun(theta):
        # 2nd argument is the success flag, 1st is the numerical cost
        return cost_and_status(theta)[0]
    
    @jax.jit
    def cost_and_jacobian(theta):
        cost, success = cost_and_status(theta)
        cost_jac = jax.jacfwd(J_cost_fun)(theta)
        return cost, cost_jac, success
    
    # convert output types from JAX to SciPy
    def obj_fun_and_grad(theta):
        # JAX input -> NumPy output
        cost, jac, success = cost_and_jacobian(jnp.asarray(theta))
        
        if not success:
            raise RuntimeError(f'Forward solver failed at theta = {theta}')
        if ~jnp.isfinite(cost):
            raise RuntimeError(f'Non-finite cost function at theta = {theta}')
        if ~jnp.all(jnp.isfinite(jac)):
            raise RuntimeError(f'Non-finite cost function jacobian at theta = {theta}')
        
        return float(cost), np.asarray(jac)
    
    # initial guess: no correction needed, theta = 0 (i.e. beta = 1)
    print('Solving the optimisation problem argmin(J(theta)) to find '
          'maximum-a-posteriori (MAP) value...')
    start_time = perf_counter()
    opt = sciopt.minimize(obj_fun_and_grad, x0=np.zeros(shape=(N_theta,)), 
                          method='BFGS', jac=True)
    t_elapsed = perf_counter() - start_time
    print(f'Optimisation problem solved in {t_elapsed:.2f} s')
    aux.check_optimiser_results(opt)
    
    theta_MAP = opt.x
    _, sol_MAP, max_res_MAP, success_MAP = run_solver_case(None, theta_MAP, solver_fun, 
                                            beta_fun, nu, x_mesh, t_mesh,
                                            u_IC, u0, uL, STEADY, out_dir=None)
    beta_r = beta_fun(theta_MAP, u=sol_MAP)
    
    aux.save_sampling(out_file, 'MAP', comprex_opts, theta_MAP, sol_MAP, 
                      max_res_MAP, success_MAP, beta_r, key_random=None)
    
    if CHECK and beta_exact is not None:
        post.compare_exact_beta(beta_r, beta_exact['beta'], 
                                x_mesh, t_mesh, beta_exact['x'], beta_exact['t'], 
                                t_plot=beta_exact['t'][-1], fig_file=out_dir/'checkBeta', 
                                PLOT=PLOT, VIDEO=VIDEO)
    
    
    #%% CHECK MAP SOLUTION VS BASELINE
    
    solver_wrap = lambda theta: run_solver_case(None, theta, solver_fun, beta_fun, 
                                                nu, x_mesh, t_mesh, u_IC, u0, uL, 
                                                STEADY, out_dir=None)
    # baseline solution: beta = 1 => theta = 0
    fiml.compare_baseline_MAP_solutions(sol_MAP, jnp.zeros_like(theta_MAP), solver_wrap, 
                                        sol_exp, x=x_mesh, x_exp=x_exp, t=t_mesh, t_exp=t_exp,
                                        PLOT=PLOT)

    
    #%% CALCULATE COVARIANCE OF THE POSTERIOR (HESSIAN)
    
    print('Calculating the (approximate) posterior covariance matrix as inverse of the hessian matrix @ MAP...')
    match hessian.lower():
        case 'full':
            H_MAP = jax.jacfwd(jax.jacfwd(J_cost_fun))(theta_MAP)
        case 'gauss_newton':
            # h_theta returns the solution and the success flag => pass a wrapper
            # function that returns only the solution, to avoid having JAX
            # differentiating w.r.t. the flag
            H_MAP = fiml.Gauss_Newton_hessian(lambda theta: h_theta(theta)[0], 
                                              theta_MAP, sigma2_obs, sigma2_prior)
        case 'test':
            aux.warning('Calculating the Hessian matrix using all available options...')
            H_MAP_full = jax.jacfwd(jax.jacfwd(J_cost_fun))(theta_MAP)
            H_MAP_GN = fiml.Gauss_Newton_hessian(lambda theta: h_theta(theta)[0], 
                                                 theta_MAP, sigma2_obs, sigma2_prior)
            err_GN = jnp.linalg.norm(H_MAP_GN - H_MAP_full) / jnp.linalg.norm(H_MAP_full)
            aux.warning(f'Gauss-Newton vs Exact Frobenius-norm error: {err_GN:.5e}')
            H_MAP = H_MAP_full
        case _:
            raise ValueError(f"Unknown Hessian matrix option '{hessian}'")
    aux.check_hessian(H_MAP)
    # rather than inverting H with inv(H), calculate is solving the linear
    # system H*H_inv = I
    C_theta_MAP = jnp.linalg.solve(H_MAP, jnp.eye(theta_MAP.size))
    
    
    #%% EVALUATE A-POSTERIORI DISTRIBUTION
    
    print(f'Drawing {N_sampling_post} samples of the {N_theta}-components theta vector a-posteriori...')
    key_random = aux.generate_key(SEED)
    theta_post_samples = fiml.posterior_sampling(theta_MAP, C_theta_MAP, N_sampling_post, key_random)
    
    solution_posterior, max_res_posterior, success_posterior = fiml.launch_sampling(theta_post_samples, run_solver_case, solver_fun, N_x, N_t, 
                    N_sampling_post, (beta_fun, nu, x_mesh, t_mesh, u_IC, u0, uL, STEADY), RUN_PARALLEL=RUN_PARALLEL, N_WORKERS=N_WORKERS)
    beta_posterior = jax.vmap(lambda theta, u: beta_fun(theta, u=u), 
                              in_axes=(1, 2), out_axes=2
                              )(theta_post_samples, solution_posterior)
    
    aux.save_sampling(out_file, 'posterior', comprex_opts, theta_post_samples, 
                      solution_posterior, max_res_posterior, success_posterior,
                      beta_posterior, jax.random.key_data(key_random))
    
    solution_posterior, max_res_posterior = fiml.handle_failed_runs(
                                                (solution_posterior, max_res_posterior), 
                                                success_posterior, 
                                                strategy='remove')
    N_sampling_posterior = jnp.count_nonzero(success_posterior)
    if N_sampling_posterior == 0:
        raise RuntimeError('All simulations failed during posterior sampling, '
                           'no data to evaluate prior distribution')
    
    
    #%% CHECK POSTERIOR DISTRIBUTION AROUND MAP POINT
    
    post.check_experiments_in_sigma_band(solution_posterior, sol_exp, x_mesh, 
                                         t_mesh, x_exp, t_exp, N_sigma=2, 
                                         t_plot=t_exp[IDX_TIME_PLOT], 
                                         fig_file=out_dir/'meanSigmaExpPosterior', PLOT=PLOT)
    
    post.compare_prior_posterior(solution_corrected, solution_posterior, sol_exp, 
                                 x_mesh, t_mesh, x_exp, t_exp, N_sigma=2, 
                                 t_plot=t_exp[IDX_TIME_PLOT], 
                                 fig_file=out_dir/'priorVSposterior', PLOT=PLOT)
    
    
    #%% CLOSE HDF5 FILE
    
    print(f'HDF5 groups: {list(out_file.keys())}')
    out_file.close()
    
    
    # all figures are created with block=False to avoid blocking the execution
    # => block the last one, so that the user can inspect them before closing 
    # them and allowing the function to return
    if PLOT:
        aux.warning('Close all figures to finish the execution')
        plt.show(block=True)


#%% POSTPROCESSING

def postprocessing(sol_file, exp_file, out_dir, beta_exact=None):
    sol_file = Path(sol_file)
    print(f'Reading FIML solution from \n\t{sol_file.resolve()}')
    # https://stackoverflow.com/a/41586571/17220538
    with h5py.File(sol_file, 'r') as file:
        print('Available groups (left) and their datasets (right): ')
        for group_name, group in file.items():
            print('\t', group_name, '\t', list(group.keys()))
        prior_data = aux.load_dataset(file, 'prior')
        posterior_data = aux.load_dataset(file, 'posterior')
        MAP_data = aux.load_dataset(file, 'MAP')
        mesh = aux.load_dataset(file, 'mesh')
    
    exp_file = Path(exp_file)
    print(f'Reading experimental data from \n\t{exp_file.resolve()}')
    with h5py.File(exp_file, 'r') as file:
        exp_data = aux.load_dataset(file)
    
    # check where experimental data lie within the +/- 2 sigma band around the
    # sample mean value, for both the prior and the posterior
    check_sigma_opts = {'exp': jnp.asarray(exp_data['solution']),
                        'x': jnp.asarray(mesh['x']),
                        't': jnp.asarray(mesh['t']),
                        'x_exp': jnp.asarray(exp_data['x']),
                        't_exp': jnp.asarray(exp_data['t']),
                        'N_sigma': 2,
                        't_plot': exp_data['t'][-1]
                        }
    post.check_experiments_in_sigma_band(prior_data['solution'], **check_sigma_opts,
                                         fig_file=out_dir/'meanSigmaExpPrior', )
    post.check_experiments_in_sigma_band(posterior_data['solution'], **check_sigma_opts,
                                         fig_file=out_dir/'meanSigmaExpPosterior')
    
    # compare prior, posterior and MAP beta correction to the exact one
    # (if known)
    if beta_exact is not None:
        post.compare_exact_beta((prior_data['beta'], posterior_data['beta'], MAP_data['beta']),
                                beta_exact['beta'], 
                                mesh['x'], mesh['t'], beta_exact['x'], beta_exact['t'], 
                                t_plot=beta_exact['t'][-1], fig_file=out_dir/'checkBeta',
                                beta_labels=('Prior', 'Posterior', 'MAP')
                                )
    
    # check if the parametrisation is able to reproduce the exact beta
    
    
    # keep the terminal on hold to avoid closing figures immediately
    aux.warning('Close all figures to finish the execution')
    plt.show(block=True)
