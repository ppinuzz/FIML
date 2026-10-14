#!/usr/bin/env python3
"""
Apply FIML methodology to Burgers' equation.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from burgers import solve_Burgers
from os import process_cpu_count
import sys
from scipy.interpolate import interp1d
import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.random as jrn
import jax.scipy.optimize as jsopt
import scipy.optimize as sciopt
import cloudpickle


FIML_pkg = Path(r'C:\Andrea\Dottorato\FIML')
if str(FIML_pkg) not in sys.path:
    sys.path.append(str(FIML_pkg))
import fiml.fiml as fiml
import fiml.auxiliaries as aux
import fiml.parallel as parallel


def run_solver_case(run_id, theta_r, N_basis_x, N_basis_t, B_x, B_x_interf, 
                    B_t, nu, x_mesh, t_mesh, u_IC, u0, uL, out_dir, BETA_MIN=1e-5):
    
    beta_r = fiml.beta(theta_r, B_x, B_t)
    # TODO: find a way to avoid having nu < 0 (i.e. not having beta < 0)
    beta_r = jnp.maximum(beta_r, BETA_MIN)
    
    # apply ICs and BCs + solve corrected Burgers' eq.
    solution = jnp.zeros(shape=(t_mesh.size, x_mesh.size))
    solution = solution.at[0,:].set(u_IC)
    solution = solution.at[:,0].set(u0)
    solution = solution.at[:,-1].set(uL)
    solution, max_res = solve_Burgers(solution, x_mesh, t_mesh, nu, beta=beta_r, WRITE_LOG=False)
    
    # SAVE DATA
    # (if out_dir is None, this function has probably been called by the
    # optimiser => no need to write data to a file)
    if out_dir is not None:
        # 0-padded at the beginning, to have '0005' rather than '5' (using '04d')
        # (using 08d you can store up to 10e6 samples)
        filename = out_dir / f'corrected_{run_id:08d}.npz'
        np.savez_compressed(filename, solution=solution, max_res=max_res)
    
    return run_id, solution, max_res



#%% USER INPUT

sol_dir = Path('solution')

# "exact case": variable viscosity, very fine mesh
exact_case = 'variable_nu.npz'

# probes data are extract from the exact case, mimicking the physical experiment
probes_dir = Path('probes')
exp_data_file = probes_dir / exact_case

# case settings
cfg_file = sol_dir / 'settings.npz'

out_dir = Path('samples_corrected')
out_dir.mkdir(exist_ok=True)

VIDEO = True

# number of basis functions (B-splines here) used in 
#   beta = 1 + sum_m sum_q theta_mq Bm(x) Bq(t)
N_basis_x = 5
N_basis_t = 7

# variance of the prior distribution (very low => data are exact truth)
# NB: this is sigma^2 (variance), NOT sigma (standard deviation)!
sigma2_prior = 1e-2
# number of drawing from the prior probabiliy distribution (should be high)
N_sampling_prior = 20
# seed used to generate random numbers (set to None to use a random seed, set
# to any value to have a predictable outcome)
SEED = 123
# for the observational covariance matrix Cm
sigma2_obs = 1e-2


RUN_PARALLEL = True
# max number of workers in the pool, each one handling a job
N_WORKERS = 4

# if true, assume data are provided
SKIP_PRIOR_SAMPLING = True


#%% ------------------ DO NOT TOUCH ANYTHING BELOW THIS LINE ------------------

# run only 3 time instants, t0, t+dt, t0+2dt, to check if everything works
TEST_RUN = False

# exact solution data are saved as [t^k, max residual, u_1, ..., u_1000], so
# actual u data starts from j = 2
#IDX_SOL_EXACT = 2
# probes data are saved as [t^k, u_1, ..., u_N] => actual data from j = 1
#IDX_SOL_PROBES = 1
# time index to plot
IDX_TIME_PLOT = -1


#%% READ DATA

# "experimental data" from probes
exp_data = np.load(exp_data_file)

# time and mesh data
x_mesh = jnp.linspace(0, 1, 1001)
t_mesh = jnp.linspace(0, 1, 201)

if TEST_RUN:
    t_mesh = t_mesh[0:3]

cfg = np.load(cfg_file)

mesh = {'x': x_mesh, 
        't': t_mesh
        }
BC = {'u0': cfg['u0'], 
      'uL': cfg['uL'], 
      'u_IC': cloudpickle.loads(cfg['u_IC'])
      }


def main(exp_data, mesh, BC, nu, out_dir, VIDEO, N_basis_x, N_basis_t, sigma_prior2, 
         N_sampling_prior, SEED, sigma2_obs, N_WORKERS, IDX_TIME_PLOT, RUN_PARALLEL,
         CHECK):
    
    # MESH DATA
    x_mesh = jnp.asarray(mesh['x'])
    t_mesh = jnp.asarray(mesh['t'])
    N_x = x_mesh.size
    N_t = t_mesh.size
    L = x_mesh[-1] - x_mesh[0]
    T = t_mesh[-1] - t_mesh[0]
    # HACK: assuming uniform spacing in time and space
    dx = x_mesh[1] - x_mesh[0]
    
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
    theta_prior = 0
    key_random = jrn.key(SEED)
    # each column of the matrix is one draw of the theta vector, which contains
    # the N_theta parameters of the beta(x,t) function
    print(f'Drawing {N_sampling_prior} samples of the {N_theta}-components theta vector...')
    # extract from normal N(0,1), move to Gaussian N(mu,sigma^2)
    theta_prior_samples = (theta_prior 
                               + jnp.sqrt(sigma_prior2) * jrn.normal(
                                   key_random, shape=(N_theta, N_sampling_prior))
                        )
    
    
    #%% SOLVER CORRECTED BURGERS' EQUATION
    
    solution_corrected = jnp.zeros(shape=(N_t, N_x, N_sampling_prior))
    # 2nd dimension 1 is not really needed, but that way they have the same
    # calling scheme (time, space, sample)
    max_res_corrected = jnp.zeros(shape=(N_t, 1, N_sampling_prior))
    if RUN_PARALLEL:
        print('-------------- PARALLEL RUN --------------')
        print(f'Starting workers pool with (at most) {N_WORKERS} workers...')
        print(f'Max number of logical CPU cores this program can call: {process_cpu_count()}')
        # 2nd argument is the object looped through => 'for' loops the ROWS
        # of the matrix, but here it should loop the columns of theta_prior_samples
        # (each column is a different sample) => loop through the transpose
        solution_corrected, max_res_corrected = parallel.run_parallel(N_WORKERS, 
                                        theta_prior_samples.T, run_solver_case, 
                                        solution_corrected, max_res_corrected,
                                        N_basis_x, N_basis_t, B_x, B_x_interf, 
                                        B_t, nu, x_mesh, t_mesh, 
                                        u_IC, u0, uL, out_dir)
    else:
        print('-------------- SEQUENTIAL RUN --------------')
        for r in range(theta_prior_samples.shape[1]):
            print(f'Running sample {r}')
            theta_r = theta_prior_samples[:,r]
            _, sol_corr, max_res_corr = run_solver_case(
                        r, theta_r, N_basis_x, N_basis_t, B_x, B_x_interf, 
                        B_t, nu, x_mesh, t_mesh, u_IC, u0, uL, out_dir)
            solution_corrected = solution_corrected.at[:,:,r].set(sol_corr)
            max_res_corrected = max_res_corrected.at[:,0,r].set(max_res_corr)

    run_type = 'Parallel' if RUN_PARALLEL else 'Sequential'
    print(f'{run_type} execution finished, data have been written to \n\t{out_dir.resolve()}')
    
    
    #%% CHECK: DO THE OBSERVED DATA LIE WITHIN +/- 2*SIGMA?
    
    # for each space-time point, calculate the sample mean of the N_sampling_prior
    # solutions (1st index: space, 2nd index: time, 3rd index: sampling)
    mean_u = jnp.mean(solution_corrected, axis=2)
    sigma_u = jnp.std(solution_corrected, axis=2)
    u_upper = mean_u + 2*sigma_u
    u_lower = mean_u - 2*sigma_u
    
    # re-interpolate experimental data over simulation time grid
    u_probe_interp = lambda u_probe: jnp.interp(t_mesh, t_exp, u_probe)
    sol_exp_interp_t = jax.vmap(u_probe_interp, in_axes=1, out_axes=1)(sol_exp)
    # data are stored as [t^k, u_1^k, ..., u_N^k] => create a column vector of
    # column vectors => flatten the 2D array in column-major order (Fortran-like)
    sol_exp_t = jnp.ravel(sol_exp_interp_t.T)
    
    if CHECK:
        plt.figure()
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
    
    
    #%% FIND MAXIMUM-A-POSTERIORI BETA/THETA
    
    def h_theta_sampled(theta, N_basis_x, N_basis_t, B_x, B_x_interf, B_t, nu, 
                x_mesh, t_mesh, u_IC, u0, uL, x_probes):
        # run_id, solution, max_res
        # k-th row of solution: [t^k, u_1^k, ..., u_Nx^k] = [t^k, u(t^k)]
        _, sol, _ = run_solver_case(None, theta, N_basis_x, N_basis_t, B_x, 
                                    B_x_interf, B_t, nu, x_mesh, t_mesh,
                                    u_IC, u0, uL, out_dir=None)
        # [t^k, u_1^k, ...m u_Np^k]     Np = N. of probes
        sol_probes = aux.sample_solution_probes(sol, x_mesh, x_probes)
        # create a column vector of column vectors to gather the entire solution
        # as [u(x,t^0), ..., u(x,t^Nt)] => flatten array in column-major order 
        # (Fortran-style) 
        sol_vet = jnp.ravel(sol_probes.T)
        return sol_vet
    
    h_theta = lambda theta: h_theta_sampled(theta, N_basis_x, N_basis_t, B_x, 
                                            B_x_interf, B_t, nu, x_mesh, t_mesh, 
                                            u_IC, u0, uL, x_exp)

    @jax.jit
    def J_cost_fun(theta):
        return fiml.cost_function(theta, d=sol_exp_t, h_beta=h_theta, 
                                  Cm=sigma2_obs, Cbeta=sigma2_prior, beta_prior=theta_prior)
    @jax.jit
    def cost_and_jacobian(theta):
        cost = J_cost_fun(theta)
        cost_jac = jax.jacfwd(J_cost_fun)(theta)
        return (cost, cost_jac)
    # convert output types from JAX to SciPy
    def obj_fun_and_grad(theta):
        # JAX input -> NumPy output
        cost, jac = cost_and_jacobian(jnp.asarray(theta))
        return float(cost), np.asarray(jac)
    
    # initial guess: no correction needed, theta = 0 (i.e. beta = 1)
    opt = sciopt.minimize(obj_fun_and_grad, x0=np.ones(shape=(N_theta,)), 
                          method='BFGS', jac=True)
    if not opt.success:
        aux.warning('Failed optimisation: the optimiser could not find the '
                    'maximum-a-posteiori (MAP) value of theta by solving '
                    'theta_MAP = argmin(J(theta))')
    print(f'Cost function value at MAP point     : {opt.fun}')
    print(f'Number of optimiser iterations       : {opt.nit}')
    print(f'Number of cost function evaluations  : {opt.nfev}')
    print(f'Number of gradient evaluations       : {opt.njev}')
    theta_MAP = opt.x
    _, sol_MAP, max_res_MAP = run_solver_case(None, theta_MAP, N_basis_x, N_basis_t, 
                                B_x, B_x_interf, B_t, nu, x_mesh, t_mesh,
                                u_IC, u0, uL, out_dir=None)
    
    if CHECK:
        beta_r = fiml.beta(theta_MAP, B_x, B_t)
        
        plt.figure()
        plt.plot(x_mesh, beta_r[:,IDX_TIME_PLOT])
# =============================================================================
#         plt.fill_between(x_mesh, u_upper[IDX_TIME_PLOT,:], u_lower[IDX_TIME_PLOT,:], 
#                          alpha=0.3, color='r')
#         plt.plot(x_mesh, u_upper[IDX_TIME_PLOT,:], color='r')
#         plt.plot(x_mesh, u_lower[IDX_TIME_PLOT,:], color='r')
#         plt.plot(x_exp, sol_exp_interp_t[IDX_TIME_PLOT], '*', label='Experiment')
# =============================================================================
        plt.xlabel(r'$x$ [-]')
        plt.ylabel(r'$\beta$ [-]')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend()
        plt.title(fr'$\beta$ correction at $t = {t_mesh[IDX_TIME_PLOT]:.5f}$ s')
        plt.show(block=False)
        
        
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
    
    
    #%% SAVE BINARY DATA
    
    binary_out_file = out_dir / 'data.npz'
    print(f'Saving data to binary file \n\t{binary_out_file.resolve()}')
    np.savez(binary_out_file, x_mesh=x_mesh, t_mesh=t_mesh,
             sol_samples=solution_corrected, mean_samples=mean_u, 
             sigma_samples=sigma_u, theta_MAP=theta_MAP, sol_MAP=sol_MAP
             )


if __name__ == '__main__':
    nu0 = 1e-2
    nu = lambda x, t, u, ux, uxx: nu0
    main(exp_data, mesh, BC, nu, out_dir, VIDEO, N_basis_x, N_basis_t,
             sigma2_prior, N_sampling_prior, SEED, sigma2_obs, N_WORKERS, IDX_TIME_PLOT, RUN_PARALLEL,
             CHECK=True)
