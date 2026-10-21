#!/usr/bin/env python3
"""
Main functions of FIML methodology.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import numpy as np
import matplotlib.pyplot as plt
from os import process_cpu_count

import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.random as jrn
import scipy.optimize as sciopt
import cloudpickle

import fiml.fiml as fiml
import fiml.auxiliaries as aux
import fiml.parallel as parallel


def run_solver_case(run_id, theta_r, solver_fun, N_basis_x, N_basis_t, B_x, B_x_interf, 
                    B_t, nu, x_mesh, t_mesh, u_IC, u0, uL, out_dir, TOL=1e-5, BETA_MIN=1e-2,
                    PRINT_ERR_MSG=True):
    
    beta_r = fiml.beta(theta_r, B_x, B_t, NO_NEG=True)
    
    # apply ICs and BCs + solve corrected Burgers' eq.
    solution = jnp.zeros(shape=(t_mesh.size, x_mesh.size))
    solution = solution.at[0,:].set(u_IC)
    solution = solution.at[:,0].set(u0)
    solution = solution.at[:,-1].set(uL)
    solution, max_res = solver_fun(solution, x_mesh, t_mesh, nu, beta=beta_r, WRITE_LOG=False, tol=TOL)
    
    # SAVE DATA
    # (if out_dir is None, this function has probably been called by the
    # optimiser => no need to write data to a file)
    if out_dir is not None:
        # 0-padded at the beginning, to have '0005' rather than '5' (using '04d')
        # (using 08d you can store up to 10e6 samples)
        filename = out_dir / f'corrected_{run_id:08d}.npz'
        np.savez_compressed(filename, solution=solution, max_res=max_res)
    
    # check if solver didn't converge or produced NaN's/Inf's
    not_converged = (max_res > TOL)
    nan_inf_values = jnp.any(~jnp.isfinite(solution), axis=1) | ~jnp.isfinite(max_res)
    # jax.lax.cond() replaces if (...) when the object is a JAX object in a JIT
    # compiled function
    jax.lax.cond(jnp.any(not_converged), 
                 # jax.debug.print() replaces print() in JIT-compiled functions
                 lambda PRINT_ERR_MSG: jax.debug.print(
                         f'{sum(not_converged)/not_converged.size} time instants '
                         'did not converge for run {run_id}') if PRINT_ERR_MSG else None,
                 lambda _: None,
                 operand=None
                 )
    
    jax.lax.cond(jnp.any(nan_inf_values), 
                 lambda PRINT_ERR_MSG: jax.debug.print(
                         f'{sum(nan_inf_values)/nan_inf_values.size} time instants ' 
                         'returned a NaN/Inf solution and/or residual for run {run_id}') if PRINT_ERR_MSG else None,
                 lambda _: None,
                 operand=None
                 )
    success = ~jnp.any(not_converged | nan_inf_values)
    
    return run_id, solution, max_res, success


def main(exp_data, mesh, BC, nu, out_dir, N_basis_x, N_basis_t, sigma2_prior, 
         N_sampling_prior, N_sampling_post, SEED, sigma2_obs, N_WORKERS, RUN_PARALLEL,
         solver_fun, CHECK=False, VIDEO=False, beta_exact=None, BATCH=False,
         IDX_TIME_PLOT=-1):
    
    if BATCH:
        aux.warning('BATCH mode selected, turning off both CHECK and VIDEO flags...')
        CHECK = False
        VIDEO = False
    
    # INPUT CHECKS
    exp_data = aux.check_file_or_data_input(exp_data, 'exp_data')
    if beta_exact is not None:
        beta_exact = aux.check_file_or_data_input(beta_exact, 'beta_exact')
    
    prior_dir = out_dir / 'prior'
    posterior_dir = out_dir / 'posterior'
    prior_dir.mkdir(exist_ok=True, parents=True)
    posterior_dir.mkdir(exist_ok=True, parents=True)
    
    # MESH DATA
    x_mesh = jnp.asarray(mesh['x'])
    t_mesh = jnp.asarray(mesh['t'])
    N_x = x_mesh.size
    N_t = t_mesh.size
    L = x_mesh[-1] - x_mesh[0]
    t0 = t_mesh[0]
    T = t_mesh[-1] - t_mesh[0]
    # HACK: assuming uniform spacing in time and space
    dx = x_mesh[1] - x_mesh[0]
    dt = t_mesh[1] - t_mesh[0]
    if not jnp.all(jnp.isclose(jnp.diff(x_mesh), dx)):
        raise ValueError('Only uniform space grids are supported')
    if not jnp.all(jnp.isclose(jnp.diff(t_mesh), dt)):
        raise ValueError('Only uniform time grids are supported')
    
    # BOUNDARY CONDITIONS
    u0 = BC['u0']
    uL = BC['uL']
    u_IC_fun = BC['u_IC']                   # function u0(x)
    u_IC = jnp.asarray(u_IC_fun(x_mesh))    # numerical discretisation of u0
    
    # EXPERIMENTAL DATA
    x_exp = jnp.asarray(exp_data['x'])
    t_exp = jnp.asarray(exp_data['t'])
    sol_exp = jnp.asarray(exp_data['solution'])
    
    
    #%% PARAMETRISE CORRECTION FUNCTION BETA
    
    # matrices containing
    #   [B0(t0) ... B0(tN)]
    #   [ ...   ...   ... ]
    #   [Bn(t0) ... Bn(tN)]
    # where t can be either time (for B_t) or space (for B_x)
    print('Calculating basis function matrices B_X and B_T...')
    B_x = fiml.create_basis_functions_matrix(N_basis_x, x_mesh)
    B_t = fiml.create_basis_functions_matrix(N_basis_t, t_mesh)
    # same, but evaluated at interface nodes
    # (N nodes => N-1 interfaces => N-1 interface nodes)
    x_interf = 0.5 * (x_mesh[:-1] + x_mesh[1:])
    # specify the original domain mesh, so that its length is okay
    B_x_interf = fiml.create_basis_functions_matrix(N_basis_x, x_interf, domain=x_mesh)
    
    N_theta = N_basis_x * N_basis_t
    print(f'Number of theta parameters to find: {N_theta}')
    
    
    #%% DRAW FROM PRIOR DISTRIBUTION
    
    # beta = 1 + sum_mq theta_mq Bm(x) Bq(t)
    # beta_prior = 1 (no correction) means theta_prior = 0
    theta_prior = np.zeros(shape=(N_theta, ))
    key_random = jrn.key(SEED)
    # each column of the matrix is one draw of the theta vector, which contains
    # the N_theta parameters of the beta(x,t) function
    print(f'Drawing {N_sampling_prior} samples of the {N_theta}-components theta vector...')
    theta_prior_samples = fiml.prior_sampling(theta_prior, sigma2_prior, 
                                              N_sampling_prior, key_random)
    

    #%% SOLVE CORRECTED BURGERS' EQUATION
    
    solution_corrected, max_res_corrected, success_corrected = fiml.launch_sampling(theta_prior_samples, run_solver_case, solver_fun, N_x, N_t, 
                    N_sampling_prior, (N_basis_x, N_basis_t, B_x, B_x_interf, 
                    B_t, nu, x_mesh, t_mesh, u_IC, u0, uL), sample_dir=prior_dir, RUN_PARALLEL=RUN_PARALLEL, N_WORKERS=N_WORKERS)
    
    solution_corrected, max_res_corrected = fiml.handle_failed_runs(
                                                (solution_corrected, max_res_corrected), 
                                                success_corrected, 
                                                strategy='remove')
    N_sampling_prior = solution_corrected.shape[-1]
    if N_sampling_prior == 0:
        raise RuntimeError('All simulations failed during prior sampling, no data to '
                           'evaluate prior distribution')
        
    
    #%% CHECK: DO THE OBSERVED DATA LIE WITHIN +/- 2*SIGMA?
    
    # for each space-time point, calculate the sample mean of the N_sampling_prior
    # solutions (1st index: space, 2nd index: time, 3rd index: sampling)
    mean_u = jnp.mean(solution_corrected, axis=2)
    sigma_u = jnp.std(solution_corrected, axis=2)
    u_upper = mean_u + 2*sigma_u
    u_lower = mean_u - 2*sigma_u
    
    if CHECK:
        # re-interpolate experimental data over simulation time grid
        u_probe_interp = lambda u_probe: jnp.interp(t_mesh, t_exp, u_probe)
        sol_exp_interp_t = jax.vmap(u_probe_interp, in_axes=1, out_axes=1)(sol_exp)
        # data are stored as [t^k, u_1^k, ..., u_N^k] => create a column vector of
        # column vectors => flatten the 2D array in column-major order (Fortran-like)
        sol_exp_t = jnp.ravel(sol_exp_interp_t.T)
        
        fig = plt.figure()
        plt.plot(x_mesh, mean_u[IDX_TIME_PLOT,:], label='Mean $u$')
        plt.fill_between(x_mesh, u_upper[IDX_TIME_PLOT,:], u_lower[IDX_TIME_PLOT,:], 
                         alpha=0.3, color='r')
        plt.plot(x_mesh, u_upper[IDX_TIME_PLOT,:], color='r')
        plt.plot(x_mesh, u_lower[IDX_TIME_PLOT,:], color='r')
        plt.plot(x_exp, sol_exp_interp_t[IDX_TIME_PLOT], '*', label='Experiment')
        plt.xlabel('$x$ [-]')
        plt.ylabel('$u$ [-]')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend()
        plt.title(fr'Mean solution with $\pm 2\sigma$ at $t = {t_mesh[IDX_TIME_PLOT]:.5f}$ s')
        plt.show(block=False)
        aux.save_fig(fig, 'mean_prior_sigma', out_dir)
    
    
    #%% FIND MAXIMUM-A-POSTERIORI BETA/THETA
    
    def h_theta_sampled(theta, N_basis_x, N_basis_t, B_x, B_x_interf, B_t, nu, 
                x_mesh, t_mesh, u_IC, u0, uL, x_probes, t_probes):
        # run_id, solution, max_res
        # k-th row of solution: [t^k, u_1^k, ..., u_Nx^k] = [t^k, u(t^k)]
        _, sol, _, success = run_solver_case(None, theta, solver_fun, N_basis_x, 
                                             N_basis_t, B_x, B_x_interf, B_t, nu, 
                                             x_mesh, t_mesh, u_IC, u0, uL, 
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
    
    h_theta = lambda theta: h_theta_sampled(theta, N_basis_x, N_basis_t, B_x, 
                                            B_x_interf, B_t, nu, x_mesh, t_mesh, 
                                            u_IC, u0, uL, x_exp, t_exp)
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
        if ~jnp.all(jnp.isfinite(cost)):
            raise RuntimeError(f'Non-finite cost function jacobian at theta = {theta}')
        
        return float(cost), np.asarray(jac)
    
    # initial guess: no correction needed, theta = 0 (i.e. beta = 1)
    print('Solving the optimisation problem argmin(J(theta)) to find '
          'maximum-a-posteriori (MAP) value...')
    opt = sciopt.minimize(obj_fun_and_grad, x0=np.zeros(shape=(N_theta,)), 
                          method='BFGS', jac=True)
    aux.check_optimiser_results(opt)
    
    theta_MAP = opt.x
    _, sol_MAP, max_res_MAP, success_MAP = run_solver_case(None, theta_MAP, solver_fun, N_basis_x, N_basis_t, 
                                B_x, B_x_interf, B_t, nu, x_mesh, t_mesh,
                                u_IC, u0, uL, out_dir=None)
    beta_r = fiml.beta(theta_MAP, B_x, B_t, NO_NEG=True)
    
    
    if CHECK and beta_exact is not None:
        # interpolate exact beta at required time instant...
        beta_interp_t = jax.vmap(lambda beta_ex: jnp.interp(t_mesh, beta_exact['t'], beta_ex), 
                                 in_axes=1, out_axes=1
                                 )(beta_exact['beta'])
        # and re-interpolate it over the actual mesh
        beta_interp_xt = jax.vmap(lambda beta_t: jnp.interp(x_mesh, beta_exact['x'], beta_t), 
                                  in_axes=0, out_axes=0
                                  )(beta_interp_t)
        
        fig = plt.figure()
        plt.plot(x_mesh, beta_r[:,IDX_TIME_PLOT], label=r'Calculated $\beta(x,t)$')
        plt.plot(x_mesh, beta_interp_xt[IDX_TIME_PLOT,:], label=r'Exact $\beta(x,t, u, u_x)$')
        plt.xlabel(r'$x$ [-]')
        plt.ylabel(r'$\beta$ [-]')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend()
        plt.title(fr'$\beta$ correction at $t = {t_mesh[IDX_TIME_PLOT]:.5f}$ s')
        plt.show(block=False)
        aux.save_fig(fig, 'check_beta', out_dir)
        
        if VIDEO:
            fig, ax = plt.subplots()
            line_beta, = ax.plot(x_mesh, jnp.zeros_like(x_mesh), label=r'$\nu = \nu_0$')
            ax.legend(loc='upper right')
            ax.set_xlim(x_mesh[0], x_mesh[-1])
            ax.set_ylim(0, 2)
            # enable interactive mode
            plt.ion()
            plt.show(block=False)
            for j in range(t_mesh.shape[0]):
                # stop animation if the user closed the window
                if not plt.fignum_exists(fig.number):
                    break
                line_beta.set_ydata(beta_r[:,j])
                fig.canvas.draw_idle()
                #fig.canvas.flush_events()
                #plt.pause(0.0000001)
                plt.pause(0.01)
                ax.set_title(f'$t = {t_mesh[j]:.4f}$ s')
            plt.ioff()
            # show figure only if it hasn't been closed by the user
            if plt.fignum_exists(fig.number):
                plt.show()
    
    
    #%% CHECK MAP SOLUTION VS BASELINE
    
    solver_wrap = lambda theta: run_solver_case(None, theta, solver_fun, 
                                    N_basis_x, N_basis_t, B_x, B_x_interf, B_t, 
                                    nu, x_mesh, t_mesh, u_IC, u0, uL, out_dir=None)
    # baseline solution: beta = 1 => theta = 0
    fiml.compare_baseline_MAP_solutions(sol_MAP, jnp.zeros_like(theta_MAP), solver_wrap, 
                                        sol_exp, x=x_mesh, x_exp=x_exp, t=t_mesh, t_exp=t_exp,
                                        PLOT=True)

    
    #%% CALCULATE COVARIANCE OF THE POSTERIOR (HESSIAN)
    
    print('Calculating the (approximate) posterior covariance matrix as inverse of the hessian matrix @ MAP...')
    H_MAP = jax.jacfwd(jax.jacfwd(J_cost_fun))(theta_MAP)
    aux.check_hessian(H_MAP)
    # rather than inverting H with inv(H), calculate is solving the linear
    # system H*H_inv = I
    C_theta_MAP = jnp.linalg.solve(H_MAP, jnp.eye(theta_MAP.size))
    
    
    #%% EVALUATE A-POSTERIORI DISTRIBUTION
    
    print(f'Drawing {N_sampling_post} samples of the {N_theta}-components theta vector a-posteriori...')
    key_random = jrn.key(SEED)
    theta_post_samples = fiml.posterior_sampling(theta_MAP, C_theta_MAP, N_sampling_post, key_random)
    
    solution_posterior, max_res_posterior, success_posterior = fiml.launch_sampling(theta_post_samples, run_solver_case, solver_fun, N_x, N_t, 
                    N_sampling_post, (N_basis_x, N_basis_t, B_x, B_x_interf, 
                    B_t, nu, x_mesh, t_mesh, u_IC, u0, uL), sample_dir=posterior_dir, RUN_PARALLEL=RUN_PARALLEL, N_WORKERS=N_WORKERS)
    
    solution_posterior, max_res_posterior = fiml.handle_failed_runs(
                                                (solution_posterior, max_res_posterior), 
                                                success_posterior, 
                                                strategy='remove')
    N_sampling_posterior = jnp.count_nonzero(success_posterior)
    if N_sampling_posterior == 0:
        raise RuntimeError('All simulations failed during posterior sampling, '
                           'no data to evaluate prior distribution')
    
    
    #%% CHECK POSTERIOR DISTRIBUTION AROUND MAP POINT
    
    # (1st index: space, 2nd index: time, 3rd index: sampling)
    mean_u_post = jnp.mean(solution_posterior, axis=2)
    sigma_u_post = jnp.std(solution_posterior, axis=2)
    u_upper_post = mean_u_post + 2*sigma_u_post
    u_lower_post = mean_u_post - 2*sigma_u_post
    
    if CHECK:
        fig = plt.figure()
        plt.plot(x_mesh, mean_u_post[IDX_TIME_PLOT,:], label='Mean $u$')
        plt.fill_between(x_mesh, u_upper_post[IDX_TIME_PLOT,:], u_lower_post[IDX_TIME_PLOT,:], 
                         alpha=0.3, color='r')
        plt.plot(x_mesh, u_upper_post[IDX_TIME_PLOT,:], color='r')
        plt.plot(x_mesh, u_lower_post[IDX_TIME_PLOT,:], color='r')
        plt.plot(x_exp, sol_exp_interp_t[IDX_TIME_PLOT], '*', label='Experiment')
        plt.xlabel('$x$ [-]')
        plt.ylabel('$u$ [-]')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend()
        plt.title(fr'Mean a-posteriori solution with $\pm 2\sigma$ at $t = {t_mesh[IDX_TIME_PLOT]:.5f}$ s')
        plt.show(block=False)
        aux.save_fig(fig, 'mean_posterior_sigma', out_dir)
        
        fig = plt.figure()
        plt.plot(x_mesh, mean_u[IDX_TIME_PLOT,:], label='Mean $u$ a priori', color='b')
        plt.fill_between(x_mesh, u_upper[IDX_TIME_PLOT,:], u_lower[IDX_TIME_PLOT,:], 
                         alpha=0.3, color='b')
        plt.plot(x_mesh, u_upper[IDX_TIME_PLOT,:], color='b')
        plt.plot(x_mesh, u_lower[IDX_TIME_PLOT,:], color='b')
        plt.plot(x_mesh, mean_u_post[IDX_TIME_PLOT,:], label='Mean $u$ a posteriori', color='r')
        plt.fill_between(x_mesh, u_upper_post[IDX_TIME_PLOT,:], u_lower_post[IDX_TIME_PLOT,:], 
                         alpha=0.3, color='r')
        plt.plot(x_mesh, u_upper_post[IDX_TIME_PLOT,:], color='r')
        plt.plot(x_mesh, u_lower_post[IDX_TIME_PLOT,:], color='r')
        plt.plot(x_exp, sol_exp_interp_t[IDX_TIME_PLOT], 'k*', label='Experiment')
        plt.xlabel('$x$ [-]')
        plt.ylabel('$u$ [-]')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend()
        plt.title(fr'Mean with $\pm 2\sigma$ at $t = {t_mesh[IDX_TIME_PLOT]:.5f}$ s')
        plt.show(block=False)
        aux.save_fig(fig, 'prior_vs_posterior', out_dir)
    
    
    #%% SAVE BINARY DATA
    
    binary_out_file = out_dir / 'data.npz'
    print(f'Saving data to binary file \n\t{binary_out_file.resolve()}')
    np.savez(binary_out_file, x=x_mesh, t=t_mesh,
             prior=solution_corrected, mean_samples=mean_u, 
             sigma_samples=sigma_u, theta_MAP=theta_MAP, sol_MAP=sol_MAP,
             beta=beta_r)
