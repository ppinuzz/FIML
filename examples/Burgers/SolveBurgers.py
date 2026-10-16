#!/usr/bin/env python3
"""
Solve Burgers' equation
    u_t + uu_x = (nu(x,t) u_x)_x     for x in [0,L] and t in [0,T]
    u(x,0) = sin(pi*x/L)
    u(0,t) = 0
    u(L,t) = 0

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

from numpy import savez_compressed
import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
import inspect
from timeit import default_timer as timer
from datetime import timedelta
import sys
import cloudpickle
import burgers


#%% USER INPUT

VIDEO = False
SOLVE = False
EXTRACT_PROBES = False

# space-time domain size
L = 1
T = 1
nu0 = 1e-2

x0 = 0
xL = L

# discretisation steps
dt = 3.125e-4
dx = 7.813e-5

# initial and boundary conditions
u_IC = lambda x: jnp.sin(jnp.pi*x/L)
u0 = 0
uL = 0

# actual viscosity 
# FIXME: current signature contains also u_xx, but it's NOT implemented a 
# dependence on u_xx, as the Jacobian wouldn't be tridiagonal => use GMRES and
# a different solver (currently not implemented)
#nu = lambda x, t, u, ux, uxx: nu0 * (1 + 20*jnp.sin(jnp.pi*x/L)*(1-jnp.exp(-t)))
nu = lambda x, t, u, ux, uxx: nu0 * (1 + 20*jnp.sin(jnp.pi*x/L)*(1-jnp.exp(-jnp.abs(u))) + 0.5*jnp.abs(ux))
# dummy function for constant viscosity
nu0_fun = lambda x, t, u, ux, uxx: nu0

# beta correction
beta = lambda x, t, u, ux, uxx: nu(x, t, u, ux, uxx) / nu0_fun(x, t, u, ux, uxx)

# probes settings
N_probes = 10
dx_in_out = 0.05
x_probes = jnp.linspace(dx_in_out, L-dx_in_out, N_probes)


#%% ------------------ DO NOT TOUCH ANYTHING BELOW THIS LINE ------------------

out_dir = Path('solution')
sol_file_basic = out_dir / 'basic.npz'
sol_file_var_nu = out_dir / 'variable_nu.npz'
beta_file = out_dir / 'beta.npz'

probes_dir = Path('probes')
probes_file_basic = probes_dir / 'basic.npz'
probes_file_var_nu = probes_dir / 'variable_nu.npz'

cfg_file = out_dir / Path('settings.npz')

out_dir.mkdir(exist_ok=True)
probes_dir.mkdir(exist_ok=True)


#%% SET CONTROL FLOW FLAGS

if len(sys.argv) > 1:
    for j in range(1, len(sys.argv)):
        match sys.argv[j]:
            case '--solve':
                SOLVE = True
            case '--probes':
                EXTRACT_PROBES = True
            case '--video':
                VIDEO = True
            case '--help':
                print('Possible flags: \n'
                      '\t --solve: solve Burgers eq., with constant and variable viscosity \n'
                      '\t --probes: sample solution at probe locations \n'
                      '\t --video: show "video" of time-dependent solution \n'
                      '\t --help: print this help')
                quit()
            case _:
                raise ValueError('Unknown CLI input')


#%% MESH

# N intervals => N + 1 points
N_x = int(jnp.ceil((xL-x0)/dx)) + 1
N_t = int(jnp.ceil(T/dt)) + 1
x_mesh = jnp.linspace(x0, xL, N_x)
t_mesh = jnp.linspace(0, T, N_t)
sol_basic = jnp.zeros(shape=(N_t, N_x))
sol_var_nu = jnp.zeros(shape=(N_t, N_x))


#%% SOLVE THE EQUATION

if SOLVE:
    # apply ICs and BCs
    sol_basic = sol_basic.at[0,:].set(u_IC(x_mesh))
    sol_basic = sol_basic.at[:,0].set(u0)
    sol_basic = sol_basic.at[:,-1].set(uL)
    sol_var_nu = sol_var_nu.at[0,:].set(u_IC(x_mesh))
    sol_var_nu = sol_var_nu.at[:,0].set(u0)
    sol_var_nu = sol_var_nu.at[:,-1].set(uL)
    
    # BASIC BURGERS' EQUATION
    start_time = timer()
    sol_basic, max_res_basic = burgers.solve_Burgers(sol_basic, x_mesh, t_mesh, nu=nu0_fun)
    end_time = timer()
    time_basic = end_time - start_time
    print(f'Elapsed time: {time_basic} s ({timedelta(seconds=time_basic)} hours)')
    
    print(f'Saving solution to \n\t{sol_file_basic.resolve()}')
    # obtain a string representation of the lambda function used for u(x,0):
    #   'u_IC = lambda x: jnp.sin(jnp.pi*x/L)'
    header_IC = inspect.getsourcelines(u_IC)[0][0].strip()
    header = ('Solution of Burgers equation with constant viscosity\n'
              f'x in [0,1], t in [0, {T}], nu0 = {nu0} m2/s\n'
              f'{header_IC}\n'
              'x [m], t [s], max_res [m/s2], u [m/s]')
    savez_compressed(sol_file_basic, x=x_mesh, t=t_mesh, L=L, T=T, 
                        nu=cloudpickle.dumps(nu0_fun), uL=uL, u0=u0, 
                        u_IC=cloudpickle.dumps(u_IC), max_res=max_res_basic, 
                        solution=sol_basic, description=header)
    
    #%% VARIABLE VISCOSITY BURGERS' EQUATION
    start_time = timer()
    sol_var_nu, max_res_var_nu = burgers.solve_Burgers(sol_var_nu, x_mesh, t_mesh, nu=nu)
    end_time = timer()
    time_var_nu = end_time - start_time
    print(f'Elapsed time: {time_var_nu} s ({timedelta(seconds=time_var_nu)} hours)')
    
    print(f'Saving solution to \n\t{sol_file_var_nu.resolve()}')
    header_IC = inspect.getsourcelines(u_IC)[0][0].strip()
    header = ('Solution of Burgers equation with variable viscosity\n'
              f'x in [0,1], t in [0, {T}], nu = nu(x,t) m2/s, nu0 = {nu0} m2/s\n'
              f'{header_IC}\n'
              'x [m], t [s], max_res [m/s2], u [m/s]')
    savez_compressed(sol_file_var_nu, x=x_mesh, t=t_mesh, L=L, T=T, 
                        nu=cloudpickle.dumps(nu), uL=uL, u0=u0, 
                        u_IC=cloudpickle.dumps(u_IC), max_res=max_res_var_nu, 
                        solution=sol_var_nu, description=header)
    
    
    #%% CASE SETTINGS
    
    savez_compressed(cfg_file, x=x_mesh, t=t_mesh, L=L, T=T, 
                     nu=cloudpickle.dumps(nu0_fun), uL=uL, u0=u0, 
                     u_IC=cloudpickle.dumps(u_IC))
    
    
    #%% EVALUATE EXACT BETA CORRECTION (for reference)
    
    ux = jnp.gradient(sol_var_nu, x_mesh, axis=1)
    # jnp.gradient() is 1st order at the boundaries => manually calculate the
    # 2nd order derivatives
    dx = x_mesh[1] - x_mesh[0]
    ux = ux.at[:, 0].set(
            (-3*sol_var_nu[:, 0] + 4*sol_var_nu[:, 1] - sol_var_nu[:, 2]) / (2*dx)
        )
    ux = ux.at[:, -1].set(
            (3*sol_var_nu[:, -1] - 4*sol_var_nu[:, -2] + sol_var_nu[:, -3]) / (2*dx)
        )
    beta_eval = jax.vmap(
                    lambda t, u, ux_t: beta(x_mesh, t, u, ux_t, None)
                    )(t_mesh, sol_var_nu, ux)
    savez_compressed(beta_file, x=x_mesh, t=t_mesh, u=sol_var_nu, ux=ux, uxx=None,
                     nu_exact=cloudpickle.dumps(nu), 
                     nu_approx=cloudpickle.dumps(nu0_fun), 
                     beta=beta_eval)
else:
    data_basic = jnp.load(sol_file_basic)
    t_mesh = data_basic['t']
    x_mesh = data_basic['x']
    max_res_basic = data_basic['max_res']
    sol_basic = data_basic['solution']
    data_var_nu = jnp.load(sol_file_var_nu)
    # HACK: assuming same time and space mesh
    max_res_var_nu = data_var_nu['max_res']
    sol_var_nu = data_var_nu['solution']
    data_beta = jnp.load(beta_file)
    beta_eval = data_beta['beta']


#%% EXTRACT "PROBES DATA"

if EXTRACT_PROBES:
    sol_probes_basic = jnp.zeros(shape=(N_t, N_probes))
    sol_probes_var_nu = jnp.zeros(shape=(N_t, N_probes))
    for j in range(N_t):
        sol_probes_basic = sol_probes_basic.at[j,:].set(jnp.interp(x_probes, x_mesh, sol_basic[j,:]))
        sol_probes_var_nu = sol_probes_var_nu.at[j,:].set(jnp.interp(x_probes, x_mesh, sol_var_nu[j,:]))
    
    print(f'Saving sampled basic solution to \n\t{probes_file_basic.resolve()}')
    header_probes = ', '.join([f'{x_probes[i]}' for i in range(0, N_probes)])
    header = ('Solution of Burgers equation with constant viscosity sampled along the axis\n'
              f'x in [0,1], t in [0, {T}], nu0 = {nu0} m2/s\n'
              f'Each row contains {N_probes} probe values\n'
              f'Probe locations [m]: {header_probes}\n'
              f'x [m], t [s], u [m/s]')
    savez_compressed(probes_file_basic, t=t_mesh, x=x_probes, 
                     solution=sol_probes_basic, description=header)
    
    print(f'Saving sampled variable viscosity solution to \n\t{probes_file_var_nu.resolve()}')
    header=('Solution of Burgers equation with variable viscosity sampled along the axis\n'
            f'x in [0,1], t in [0, {T}], nu = nu(x,t) m2/s\n'
            f'Each row contains {N_probes} probe values\n'
            f'Probe locations [m]: {header_probes}\n'
            f'x [m], t [s], u [m/s]')
    savez_compressed(probes_file_var_nu, t=t_mesh, x=x_probes, 
                     solution=sol_probes_var_nu, description=header)


#%% PLOT SOLUTION

if SOLVE:
    plt.figure()
    plt.plot(x_mesh, u_IC(x_mesh), label='$u(x,0)$')
    plt.plot(x_mesh, sol_basic[-1,:], label=r'$u$ with $\nu = \nu_0$')
    plt.plot(x_mesh, sol_var_nu[-1,:], label=r'$u$ with $\nu = \nu(x,t,u,u_x)$')
    plt.xlabel('x [m]')
    plt.ylabel('u [m/s]')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.legend()
    plt.title(f'$t = T = {T}$ s')
    plt.show(block=False)
    
    # =============================================================================
    # plt.figure()
    # plt.plot(x_mesh, jnp.full_like(x_mesh, nu0), label=r'$\nu = \nu_0$')
    # plt.plot(x_mesh, nu(x_mesh, T, sol_var_nu[-1,:]), label=r'$\nu = \nu(x,t,u,u_x)$')
    # plt.xlabel('x [m]')
    # plt.ylabel(r'$\nu$ [m$^2$/s]')
    # plt.autoscale(enable=True, axis='x', tight=True)
    # plt.legend()
    # plt.title(rf'$\nu(x, t={T})$')
    # plt.show(block=False)
    # =============================================================================
    
    plt.figure()
    plt.semilogy(t_mesh, max_res_basic, label=r'$\nu = \nu_0$')
    plt.semilogy(t_mesh, max_res_var_nu, label=r'$\nu = \nu(x,t,u,u_x)$')
    plt.xlabel('t [s]')
    plt.ylabel(r'$\max|\mathcal{B}(u^k)|$ [m/s$^2$]')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title('Maximum residual')
    plt.legend()
    # non blocking, so that it won't wait for the user to close the window
    plt.show(block=False)
    #plt.show()
    
    plt.show()

if VIDEO:
# =============================================================================
#     fig, ax = plt.subplots()
#     line_basic, = ax.plot(x_mesh, jnp.zeros_like(x_mesh), label=r'$\nu = \nu_0$')
#     line_var_nu, = ax.plot(x_mesh, jnp.zeros_like(x_mesh), label=r'$\nu = \nu(x,t)$')
#     ax.legend(loc='upper right')
#     ax.set_xlim(0, L)
#     ax.set_ylim(-2, 2)
#     # enable interactive mode
#     plt.ion()
#     plt.show(block=False)
#     for j in range(t_mesh.shape[0]):
#         # stop animation if the user closed the window
#         if not plt.fignum_exists(fig.number):
#             break
#         line_basic.set_ydata(sol_basic[j, :])
#         line_var_nu.set_ydata(sol_var_nu[j, :])
#         fig.canvas.draw_idle()
#         #fig.canvas.flush_events()
#         #plt.pause(0.0000001)
#         plt.pause(0.01)
#         ax.set_title(f'$t = {t_mesh[j]:.4f}$ s')
#     plt.ioff()
#     # show figure only if it hasn't been closed by the user
#     if plt.fignum_exists(fig.number):
#         plt.show()
# =============================================================================
    
    fig, ax = plt.subplots()
    line, = ax.plot(x_mesh, jnp.zeros_like(x_mesh))
    ax.legend(loc='upper right')
    ax.set_xlim(0, L)
    ax.set_ylim(beta_eval.min(), beta_eval.max())
    # enable interactive mode
    plt.ion()
    plt.show(block=False)
    for j in range(t_mesh.shape[0]):
        # stop animation if the user closed the window
        if not plt.fignum_exists(fig.number):
            break
        line.set_ydata(beta_eval[j,:])
        fig.canvas.draw_idle()
        #fig.canvas.flush_events()
        #plt.pause(0.0000001)
        plt.pause(0.01)
        ax.set_title(f'$t = {t_mesh[j]:.4f}$ s')
    plt.ioff()
    # show figure only if it hasn't been closed by the user
    if plt.fignum_exists(fig.number):
        plt.show()
