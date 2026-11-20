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
import cloudpickle
import pickle

import fiml.fiml as fiml
import fiml.auxiliaries as aux
import fiml.post as post
import fiml.beta as betamod
import fiml.objects as objs


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
    # FIXME: implemented only for steady problems...
    beta_check = beta_fun(theta_r, x=x_mesh, t=t_mesh, u=solution)
    beta_valid, beta_stats = betamod.is_valid_beta(beta_check)
    success = ~jnp.any(not_converged | nan_inf_values) & beta_valid.all()
    # jax.lax.cond() replaces if (...) when the object is a JAX object in a JIT
    # compiled function
    jax.lax.cond(PRINT_ERR_MSG & jnp.any(not_converged), 
                 # jax.debug.print() replaces print() in JIT-compiled functions
                 lambda _: jax.debug.print(
                             '{n_fail}/{n_tot} time instants did not converge for run {run_id} (n. beta < 0: {N_beta_neg}/{N_beta})',
                             n_fail=jnp.count_nonzero(not_converged),
                             n_tot=not_converged.size,
                             run_id=run_id,
                             N_beta_neg=beta_stats['N_neg'],
                             N_beta=beta_check.size),
                 lambda _: None,
                 operand=PRINT_ERR_MSG
                 )
        
    jax.lax.cond(PRINT_ERR_MSG & jnp.any(nan_inf_values), 
                 lambda _: jax.debug.print(
                             '{n_nan}/{n_tot} time instants returned a NaN/Inf '
                             'solution for run {run_id}  (n. beta < 0: {N_beta_neg}/{N_beta})',
                             n_nan=jnp.count_nonzero(nan_inf_values),
                             n_tot=nan_inf_values.size,
                             run_id=run_id,
                             N_beta_neg=beta_stats['N_neg'],
                             N_beta=beta_check.size),
                 lambda _: None,
                 operand=PRINT_ERR_MSG
                 )
    jax.lax.cond(PRINT_ERR_MSG & ~beta_valid.all(),
                 lambda _: jax.debug.print(
                             'Run {run_id}: invalid beta at {n_bad}/{n_total} points '
                             '(nonpositive: {n_neg}, nonfinite: {n_nan})',
                            run_id=run_id,
                            n_bad=beta_check.size - beta_stats['N_valid'],
                            n_total=beta_check.size,
                            n_neg=beta_stats['N_neg'],
                            n_nan=beta_stats['N_nan']
                ),
                lambda _: None,
                operand=None
                )
    
    
    return run_id, solution, max_res, success


def field_inversion(exp_data, problem, cost_fun_obj, beta_opts, analysis_obj, 
             exact_sol=None, input_file=None):
    
    optimiser_opts = analysis_obj.to_dict()['optimisation']
    run_opts = analysis_obj.options
    
    # OUTPUT DATA FILE
    out_filename = analysis_obj.files.output_dir / 'output.h5'
    # common compression options
    # shuffle: allows blocks to be moved around, possible memory gain
    # gzip: most portable and fairly fast
    # compression_opts: for GZIP, it's the compressione level (9: maximum)
    comprex_opts = {'shuffle': True, 'compression': 'gzip', 'compression_opts': 9}
    with h5py.File(out_filename, 'w') as out_file:
        # MESH DATA
        x_mesh = problem.x_mesh
        t_mesh = problem.t_mesh
        STEADY = problem.STEADY
        T = problem.T
        
        N_x = problem.Nx
        N_t = problem.Nt
        t0 = t_mesh[0]
        dt = problem.dt
        
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
        
        print(f'Saving boundary conditions to \n\t{out_filename.resolve()}...')
        hdf5_group = out_file.create_group('boundary_conditions')
        aux.save_dict_to_hdf5(problem.BC, hdf5_group)
        out_file.flush()
        
        if run_opts.save_experiments:
            # save experimental data to file using the same format as solution data
            # (for reproducibility)
            exp_out_file = h5py.File(analysis_obj.files.output_dir/'experiments.h5', 'w')
            tmp_set = exp_out_file.create_dataset(name='solution', data=exp_data.solution, **comprex_opts)
            tmp_set.attrs['description'] = ('The i-th row contains all the experimental data at the i-th time instant, '
                                            'each column is a different probe')
            tmp_set = exp_out_file.create_dataset(name='x', data=exp_data.x, **comprex_opts)
            tmp_set.attrs['description'] = 'Probes coordinates'
            tmp_set = exp_out_file.create_dataset(name='t', data=exp_data.t, **comprex_opts)
            tmp_set.attrs['description'] = 'Time instants acquired in the experiment'
            # write to disk without closing the file
            print(f'Writing experimental data to \n\t{Path(exp_out_file.filename).resolve()}')
            exp_out_file.close()
        
        if not STEADY:
            # check that experimental data and simulation data start and end at the same
            # time instant (or that, at least, the simulation is longer than the experiment)
            # discretisation with step dt => any number within -dt/2 and +dt/2 is associated
            # to the instant 0
            if exp_data.t[0] < t0-0.5*dt or exp_data.t[0] > t0+0.5*dt:
                raise ValueError('Experimental data and simulation must start at the same time instant '
                                 '(within a dt/2)\n'
                                 f'Experimental data t0  : {exp_data.t[0]} s\n'
                                 f'Simulation t0         : {t0} s\n')
            if T < exp_data.t[-1]:
                aux.warning('Numerical simulation is shorter than the experimental data: \n'
                            f'T numerical: {T} s\n'
                            f'T experiment: {exp_data.t[-1]} s\n'
                            f'Limiting experimental data to numerical simulation end time T = {T} s')
                idx_valid = (exp_data.t <= T)
                exp_data.t = exp_data.t[idx_valid]
                exp_data.solution = exp_data.solution[idx_valid, :]
        
        
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
        
        beta_fun, beta_params = betamod.parametrise_beta(beta_opts, x_mesh, t_mesh, STEADY)
        FIML = objs.FieldInversionObject(problem=problem, exp_data=exp_data,
                                         beta_fun=beta_fun, beta_params=beta_params,
                                         beta_opts=beta_opts)
        N_theta = FIML.N_params
        
        print(f'Available data points: {exp_data.solution.size}')
        print(f'Unknown parameters: {N_theta}')
        if exp_data.solution.size < N_theta:
            aux.warning(f'Under-determined system: {N_theta} to find, but only '
                        f'{exp_data.solution.size} data points')
        
        
        #%% DRAW FROM PRIOR DISTRIBUTION
        
        # default is theta prior = 0 (i.e. beta_prior = 1, no correction)
        theta_prior = cost_fun_obj.generate_theta_prior(N_theta)
        key_random = aux.generate_key(analysis_obj.seed)
        # each column of the matrix is one draw of the theta vector, which contains
        # the N_theta parameters of the beta(x,t) function
        print(f'Drawing {analysis_obj.num_samples_prior} samples of the {N_theta}-components theta vector...')
        theta_prior_samples = fiml.prior_sampling(theta_prior, cost_fun_obj.C_theta, 
                                                  analysis_obj.num_samples_prior, key_random)
        
    
        #%% SOLVE CORRECTED BURGERS' EQUATION => PRIOR DISTRIBUTION
        
        FIML.prior = fiml.launch_sampling(
                        theta_prior_samples, run_solver_case, problem.solver_fun, N_x, N_t, 
                        analysis_obj.num_samples_prior, problem.solver_args(beta_fun), 
                        RUN_PARALLEL=run_opts.run_parallel, N_WORKERS=run_opts.parallel_workers)
        FIML.prior.key_random = jax.random.key_data(key_random)
        FIML.prior.x = problem.x_mesh
        FIML.prior.t = problem.t_mesh
        
        # save both cleaned runs (e.g. 'FIML.prior.solution') and original runs,
        # with non-converged or failed runs (e.g. 'FIML.prior.solution_orig')
        aux.save_sampling(out_file, 'prior', comprex_opts, FIML.prior, save_failed_runs=True)
        
        if FIML.prior.N_valid == 0:
            raise RuntimeError('All simulations failed during prior sampling, no data to '
                               'evaluate prior distribution')
            
        
        #%% CHECK: DO THE OBSERVED DATA LIE WITHIN +/- 2*SIGMA?
        
        post.check_experiments_in_sigma_band(FIML.prior, exp_data, N_sigma=2, 
                                             t_plot=exp_data.t[run_opts.idx_time_plot], 
                                             fig_file=analysis_obj.files.output_dir/'meanSigmaExpPrior', PLOT=run_opts.plot)
        
        
        #%% FIND MAXIMUM-A-POSTERIORI BETA/THETA
        
        def h_theta(theta):
            # run_id, solution, max_res
            # k-th row of solution: [t^k, u_1^k, ..., u_Nx^k] = [t^k, u(t^k)]
            _, sol, _, success = run_solver_case(None, theta, problem.solver_fun, 
                                                 *problem.solver_args(beta_fun),
                                                 out_dir=None, PRINT_ERR_MSG=False)
            # sample numerical solution to have it at the same probe locations
            # [t^k, u_1^k, ...m u_Np^k]     Np = N. of probes
            sol_probes = aux.sample_solution(sol, x_mesh=x_mesh, t_mesh=t_mesh, 
                                             x_probes=exp_data.x, t_probes=exp_data.t)
            # create a column vector of column vectors to gather the entire solution
            # as [u(x,t^0), ..., u(x,t^Nt)] => flatten array in column-major order 
            # (Fortran-style) 
            sol_vet = jnp.ravel(sol_probes.T)
            return sol_vet, success
        
        cost_fun_obj.h_theta = h_theta
        # now that all the methods of the cost function are defined, store the 
        # object into the FIML object
        FIML.cost = cost_fun_obj
        
        @jax.jit
        def cost_with_aux(theta):
            cost, success = FIML.cost(theta)
            return cost, (cost, success)
    
        @jax.jit
        def cost_and_jacobian(theta):
            # has_aux=True specifies that the function returns (f(theta), auxiliaries)
            # and that it should be differentiated only w.r.t. the 1st output
            # (the actual numerical output)
            grad, (cost, success) = jax.jacfwd(cost_with_aux, has_aux=True)(theta)
            return cost, grad, success
    
        FIML.cost.jacobian = lambda theta: cost_and_jacobian(theta)[1]
        
        
        print('Minimising the cost function J(theta) to find maximum-a-posteriori '
              '(MAP) value...')
        aux.print_optimiser_info(optimiser_opts)
        start_time = perf_counter()
        if optimiser_opts['optimiser'].lower() == 'bfgs':
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
            opt = sciopt.minimize(obj_fun_and_grad, x0=np.zeros(shape=(N_theta,)), 
                                  method='BFGS', jac=True)
        else:
            opt = fiml.minimise_cost_function(cost_and_jacobian, x0=jnp.zeros(shape=(N_theta,)),
                                              method=optimiser_opts['optimiser'],
                                              learn_rate=optimiser_opts['learning_rate'],
                                              max_iter=optimiser_opts['max_iterations'])
        t_elapsed = perf_counter() - start_time
        print(f'Optimisation problem solved in {t_elapsed:.2f} s')
        aux.check_optimiser_results(opt)
        
        theta_MAP = opt.x
        _, sol_MAP, max_res_MAP, success_MAP = run_solver_case(None, theta_MAP, problem.solver_fun, 
                                                *problem.solver_args(beta_fun), out_dir=None)
        
        FIML.MAP = objs.MAPResult(theta=theta_MAP, solution=sol_MAP, 
                                  beta=beta_fun(theta_MAP, u=sol_MAP),
                                  max_res=max_res_MAP, success=success_MAP, opt=opt,
                                  x=problem.x_mesh, t=problem.t_mesh)
        
        aux.print_theta(FIML.MAP.theta, theta_prior, problem.STEADY, problem.t_mesh)
        
        start_time = perf_counter()
        FIML.MAP.jacobian = cost_and_jacobian(FIML.MAP.theta)[1]  # cost, jacobian, success
        t_elapsed = perf_counter() - start_time
        print(f'Time required to evaluate cost function jacobian: {t_elapsed:.3e} s')
        
        if run_opts.plot:
            post.plot_convergence_history(opt, fig_file=analysis_obj.files.output_dir/'optimiserHistory')
        
        
        #%% CHECK MAP SOLUTION VS BASELINE
        
        solver_wrap = lambda theta: run_solver_case(None, theta, problem.solver_fun,
                                                    *problem.solver_args(beta_fun), 
                                                    out_dir=None)
        # baseline solution: beta = 1 => theta = 0
        fiml.compare_baseline_MAP_solutions(FIML.MAP, jnp.zeros_like(FIML.MAP.theta), solver_wrap, 
                                            exp_data, t=t_mesh, t_exp=exp_data.t,
                                            PLOT=run_opts.plot)
    
        if run_opts.check_outputs and exact_sol is not None:
            # check if the parametrisation is able to reproduce the exact beta
            theta_ex = post.check_parametrisation_fitness(exact_sol, beta_opts,
                                                          fig_file=analysis_obj.files.output_dir/'checkBetaParam')
    
            run_solver_fun = lambda theta_ex: run_solver_case(None, theta_ex, 
                                                problem.solver_fun, 
                                                *problem.solver_args(beta_fun),
                                                out_dir=None)[1:]
            # solve the PDE with the "exact" value of theta => you should get the
            # exact solution
            post.check_exact_beta_PDE(theta_ex, run_solver_fun, exact_sol, problem)
    
        
        #%% CALCULATE COVARIANCE OF THE POSTERIOR (HESSIAN)
        
        print('Calculating the (approximate) posterior covariance matrix as inverse of the hessian matrix @ MAP...')
        start_time = perf_counter()
        match analysis_obj.hessian.lower():
            case 'full':
                # const_with_aux returns (jacobian, aux) => select 0th element
                H_MAP = jax.jacfwd(jax.jacfwd(cost_with_aux, has_aux=True), has_aux=True)(FIML.MAP.theta)[0]
            case 'gauss_newton':
                # h_theta returns the solution and the success flag => pass a wrapper
                # function that returns only the solution, to avoid having JAX
                # differentiating w.r.t. the flag
                H_MAP = fiml.Gauss_Newton_hessian(lambda theta: h_theta(theta)[0], 
                                                  FIML.MAP.theta, cost_fun_obj.C_data, cost_fun_obj.C_theta)
            case 'test':
                aux.warning('Calculating the Hessian matrix using all available options...')
                H_MAP_full = jax.jacfwd(jax.jacfwd(cost_with_aux, has_aux=True), has_aux=True)(FIML.MAP.theta)[0]
                H_MAP_GN = fiml.Gauss_Newton_hessian(lambda theta: h_theta(theta)[0], 
                                                     FIML.MAP.theta, cost_fun_obj.C_data, cost_fun_obj.C_theta)
                err_GN = jnp.linalg.norm(H_MAP_GN - H_MAP_full) / jnp.linalg.norm(H_MAP_full)
                aux.warning(f'Gauss-Newton vs Exact Frobenius-norm error: {err_GN:.5e}')
                H_MAP = H_MAP_full
            case _:
                raise ValueError(f"Unknown Hessian matrix option '{analysis_obj.hessian}'")
        t_elapsed = perf_counter() - start_time
        print(f'Time required to evaluate cost function hessian: {t_elapsed:.3e} s')
        
        FIML.MAP.hessian = H_MAP
        aux.check_hessian(FIML.MAP.hessian)
        # rather than inverting H with inv(H), calculate is solving the linear
        # system H*H_inv = I
        FIML.MAP.covariance = jnp.linalg.solve(FIML.MAP.hessian, jnp.eye(FIML.MAP.theta.size))
        
        aux.save_sampling(out_file, 'MAP', comprex_opts, FIML.MAP)
        
        
        #%% EVALUATE A-POSTERIORI DISTRIBUTION
        
        if analysis_obj.num_samples_posterior > 0:
            print(f'Drawing {analysis_obj.num_samples_posterior} samples of the {N_theta}-components theta vector a-posteriori...')
            key_random = aux.generate_key(analysis_obj.seed)
            theta_post_samples = fiml.posterior_sampling(FIML.MAP.theta, FIML.MAP.covariance, analysis_obj.num_samples_posterior, key_random)
            
            FIML.posterior = fiml.launch_sampling(
                                theta_post_samples, run_solver_case, problem.solver_fun, 
                                N_x, N_t, analysis_obj.num_samples_posterior, problem.solver_args(beta_fun),
                                RUN_PARALLEL=run_opts.run_parallel, N_WORKERS=run_opts.parallel_workers)
            FIML.posterior.key_random = jax.random.key_data(key_random)
            FIML.posterior.x = problem.x_mesh
            FIML.posterior.t = problem.t_mesh
            
            aux.save_sampling(out_file, 'posterior', comprex_opts, FIML.posterior, save_failed_runs=True)
            if FIML.posterior.N_valid == 0:
                raise RuntimeError('All simulations failed during posterior sampling, '
                                   'no data to evaluate posterior distribution')
        else:
            aux.warning('analysis_obj.num_samples_posterior <= 0, skipping posterior sampling')
        
        
        #%% CHECK POSTERIOR DISTRIBUTION AROUND MAP POINT
        
        if analysis_obj.num_samples_posterior > 0:
            post.check_experiments_in_sigma_band(FIML.posterior, exp_data, N_sigma=2, 
                                                 t_plot=exp_data.t[run_opts.idx_time_plot], 
                                                 fig_file=analysis_obj.files.output_dir/'meanSigmaExpPosterior', PLOT=run_opts.plot)
            
            post.compare_prior_posterior(FIML.prior, FIML.posterior, exp_data, MAP=FIML.MAP,
                                         N_sigma=2, t_plot=exp_data.t[run_opts.idx_time_plot], 
                                         fig_file=analysis_obj.files.output_dir/'priorVSposterior', PLOT=run_opts.plot)
            
            # see how beta correction has improved from prior to posterior
            if run_opts.check_outputs and exact_sol is not None:
                post.compare_exact_beta((FIML.prior, FIML.MAP, FIML.posterior), exact_sol, 
                                        t_plot=exact_sol.t[-1], fig_file=analysis_obj.files.output_dir/'checkBeta', 
                                        PLOT=run_opts.plot, VIDEO=run_opts.show_videos, beta_labels=('Prior', 'MAP', 'Posterior'))
        else:
            post.compare_prior_posterior(FIML.prior, None, exp_data, MAP=FIML.MAP,
                                         N_sigma=2, t_plot=exp_data.t[run_opts.idx_time_plot], 
                                         fig_file=analysis_obj.files.output_dir/'priorVSposterior', PLOT=run_opts.plot)
            
            if run_opts.check_outputs and exact_sol is not None:
                post.compare_exact_beta((FIML.prior, FIML.MAP), exact_sol, 
                                        t_plot=exact_sol.t[-1], fig_file=analysis_obj.files.output_dir/'checkBeta', 
                                        PLOT=run_opts.plot, VIDEO=run_opts.show_videos, beta_labels=('Prior', 'MAP'))
        
        
        #%% CLOSE HDF5 FILE
        
        print(f'HDF5 groups: {list(out_file.keys())}')
        out_file.close()
        
        # all figures are created with block=False to avoid blocking the execution
        # => block the last one, so that the user can inspect them before closing 
        # them and allowing the function to return
        if run_opts.plot:
            aux.warning('Close all figures to terminate the execution or generate an exit.fiml file')
            
            exit_file = Path('exit.fiml')
            def check_exit_file():
                if exit_file.is_file():
                    aux.warning('exit.fiml file detected, closinf figures and terminating execution...')
                    plt.close('all')
                    return False
                return True
    
            # timer associated with the GUI event loop
            # (GUI event loop is needed to keep track of events like closing a figure)
            fig = plt.gcf()
            timer = fig.canvas.new_timer(interval=500)  # 500 ms
            timer.add_callback(check_exit_file)
    
            # start the timer and block execution
            timer.start()
            try:
                plt.show(block=True)
            finally:
                timer.stop()
    
    
        if run_opts.save_FIML_object:
            FIML_class_file = out_filename.parent / 'FIML_object.pkl'
            print(f'Saving FIML class object to \n\t{FIML_class_file.resolve()}')
            with open(FIML_class_file, 'wb') as file:
                cloudpickle.dump(FIML, file)
    
    return FIML


def machine_learning(FIML):
    pass



#%% POSTPROCESSING

def postprocessing(FIML, exp_data, problem, out_dir, sol_exact=None):
    
    if FIML.prior.t.size == 1:
        aux.warning('Only one time instant in the time mesh, assuming steady flow...')
        STEADY = True
        IDX_TIME_PLOT = -1
    else:
        STEADY = False
    
    # check where experimental data lie within the +/- 2 sigma band around the
    # sample mean value, for both the prior and the posterior
    check_sigma_opts = {'N_sigma': 2, 't_plot': exp_data.t[IDX_TIME_PLOT]}
    # remove failed simulations
    # (use name, to unpack the tuple composed of only 1 element here)
    post.check_experiments_in_sigma_band(FIML.prior, exp_data, **check_sigma_opts, 
                                         fig_file=out_dir/'meanSigmaExpPrior')

    post.check_experiments_in_sigma_band(FIML.posterior, exp_data, **check_sigma_opts,
                                         fig_file=out_dir/'meanSigmaExpPosterior')
    
    post.compare_prior_posterior(FIML.prior, FIML.posterior, exp_data, N_sigma=2, t_plot=exp_data.t[IDX_TIME_PLOT], 
                                 fig_file=out_dir/'priorVSposterior')
    
    
    if sol_exact is not None:
        # compare prior, posterior and MAP beta correction to the exact one
         post.compare_exact_beta((FIML.prior, FIML.MAP, FIML.posterior), sol_exact, 
                                 t_plot=sol_exact.t[-1], fig_file=out_dir/'checkBeta', 
                                 beta_labels=('Prior', 'Posterior', 'MAP'))
         
         # check if the parametrisation is able to reproduce the exact beta
         theta_ex = post.check_parametrisation_fitness(sol_exact, FIML.beta_opts,
                                                       fig_file=out_dir/'checkBetaParam')

         run_solver_fun = lambda theta_ex: run_solver_case(None, theta_ex, 
                                             problem.solver_fun, 
                                             *problem.solver_args(FIML.beta_fun),
                                             out_dir=None)[1:]
         # solve the PDE with the "exact" value of theta => you should get the
         # exact solution
         post.check_exact_beta_PDE(theta_ex, run_solver_fun, sol_exact, problem)
    
    # keep the terminal on hold to avoid closing figures immediately
    aux.warning('Close all figures to terminate the execution')
    plt.show(block=True)
