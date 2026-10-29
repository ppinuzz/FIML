#!/usr/bin/env python3
"""
Solve Burgers' equation
    u_t + uu_x = (nu(x,t) u_x)_x     for x in [0,L] and t in [0,T]
    u(x,0) = u0(x)
    u(0,t) = u0
    u(L,t) = uL

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta
import sys
import cloudpickle
import h5py
import burgers
# setup files directory
setup_dir = Path(r'./setup_Burgers')
if str(setup_dir) not in sys.path:
    sys.path.append(str(setup_dir))


#%% USER INPUT

# show video of u(x,t)
VIDEO = False
# solve Burgers' equation
SOLVE = False
# read solver solution (or previously calculated solution) and save 
EXTRACT_PROBES = False

# setup files
#import setup_Burgers.BChom_ICsin_nu_xt as cfg
#import setup_Burgers.BChom_ICsin_nu_xuux as cfg
import setup_Burgers.BCDirichlet_steady_nu_u as cfg


#%% ------------------ DO NOT TOUCH ANYTHING BELOW THIS LINE ------------------

out_dir = Path('solution_Burgers') / Path(cfg.__file__).stem
sol_file = out_dir / 'solution.h5'
probes_file = out_dir / 'probes.h5'

out_dir.mkdir(exist_ok=True, parents=True)

# common compression options
# shuffle: allows blocks to be moved around, possible memory gain
# gzip: most portable and fairly fast
# compression_opts: for GZIP, it's the compressione level (9: maximum)
comprex_opts = {'shuffle': True, 'compression': 'gzip', 'compression_opts': 9}


sol_file_basic = out_dir / 'basic.npz'
sol_file_var_nu = out_dir / 'variable_nu.npz'
beta_file = out_dir / 'beta.npz'
probes_file_basic = out_dir / 'probes_basic.npz'
probes_file_var_nu = out_dir / 'probes_variable_nu.npz'
cfg_file = out_dir / Path('settings.npz')


#%% SET CONTROL FLOW FLAGS

if len(sys.argv) > 1:
    for j in range(1, len(sys.argv)):
        match sys.argv[j]:
            case '--solve' | '-s':
                SOLVE = True
            case '--probes' | '-p':
                EXTRACT_PROBES = True
            case '-sp' | '-ps':
                SOLVE = True
                EXTRACT_PROBES = True
            case '--video' | '-v':
                VIDEO = True
            case '--help' | '-h':
                print('Possible flags: \n'
                      '\t --solve, -s: solve Burgers eq., with constant and variable viscosity \n'
                      '\t --probes, -p: sample solution at probe locations \n'
                      '\t --video, -s: show "video" of time-dependent solution \n'
                      '\t --help, -h: print this help')
                quit()
            case _:
                raise ValueError('Unknown CLI input')


#%% MESH

# N intervals => N + 1 points
N_x = int(jnp.ceil((cfg.xL - cfg.x0)/cfg.dx)) + 1
x_mesh = jnp.linspace(cfg.x0, cfg.xL, N_x)
if cfg.dt is None:
    STEADY = True
    burgers.warning('The problem will be treated as steady')
    N_t = 1
else:
    STEADY = False
    N_t = int(jnp.ceil(cfg.T/cfg.dt)) + 1
t_mesh = jnp.linspace(0, cfg.T, N_t)
sol_basic = jnp.zeros(shape=(N_t, N_x))
sol_var_nu = jnp.zeros(shape=(N_t, N_x))


#%% SOLVE THE EQUATION

if SOLVE:
    # apply ICs and BCs
    if not STEADY:
        sol_basic = sol_basic.at[0,:].set(cfg.u_IC(x_mesh))
    sol_basic = sol_basic.at[:,0].set(cfg.u0)
    sol_basic = sol_basic.at[:,-1].set(cfg.uL)
    if not STEADY:
        sol_var_nu = sol_var_nu.at[0,:].set(cfg.u_IC(x_mesh))
    sol_var_nu = sol_var_nu.at[:,0].set(cfg.u0)
    sol_var_nu = sol_var_nu.at[:,-1].set(cfg.uL)
    
    # BASIC BURGERS' EQUATION
    start_time = timer()
    sol_basic, max_res_basic = burgers.solve_Burgers(sol_basic, x_mesh, t_mesh, 
                                                     nu=cfg.nu0_fun, STEADY=STEADY)
    end_time = timer()
    time_basic = end_time - start_time
    print(f'Elapsed time: {time_basic} s ({timedelta(seconds=time_basic)} hours)')
    
    
    #%% VARIABLE VISCOSITY BURGERS' EQUATION
    
    start_time = timer()
    sol_var_nu, max_res_var_nu = burgers.solve_Burgers(sol_var_nu, x_mesh, t_mesh, 
                                                       nu=cfg.nu, STEADY=STEADY)
    end_time = timer()
    time_var_nu = end_time - start_time
    print(f'Elapsed time: {time_var_nu} s ({timedelta(seconds=time_var_nu)} hours)')
    
    
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
                    lambda t, u, ux_t: cfg.beta(x_mesh, t, u, ux_t, None)
                    )(t_mesh, sol_var_nu, ux)
    
    #%% SAVE DATA
    
    # common compression options
    # shuffle: allows blocks to be moved around, possible memory gain
    # gzip: most portable and fairly fast
    # compression_opts: for GZIP, it's the compressione level (9: maximum)
    comprex_opts = {'shuffle': True, 'compression': 'gzip', 'compression_opts': 9}
    print(f'Writing solution to \n\t{sol_file.resolve()}')
    with h5py.File(sol_file, 'w') as file:
        hdf5_mesh = file.create_group('mesh')
        tmp_set = hdf5_mesh.create_dataset(name='x', data=x_mesh, **comprex_opts)
        tmp_set.attrs['description'] = 'Spatial mesh'
        tmp_set = hdf5_mesh.create_dataset(name='t', data=t_mesh, **comprex_opts)
        tmp_set.attrs['description'] = 'Time mesh'
        
        hdf5_cfg = file.create_group('settings')
        # cannot pass compression options when storing scalar values
        tmp_set = hdf5_cfg.create_dataset(name='L', data=cfg.L)
        tmp_set.attrs['description'] = 'Domain length'
        tmp_set = hdf5_cfg.create_dataset(name='T', data=cfg.T)
        tmp_set.attrs['description'] = 'Interval duration'
        tmp_set = hdf5_cfg.create_dataset(name='u0', data=cfg.u0)
        tmp_set.attrs['description'] = 'Dirichlet boundary condition u(0,t)'
        tmp_set = hdf5_cfg.create_dataset(name='uL', data=cfg.uL)
        tmp_set.attrs['description'] = 'Dirichlet boundary condition u(L,t)'
        #tmp_set = hdf5_cfg.create_dataset(name='u_IC', data=cloudpickle.dumps(cfg.u_IC))
        tmp_set = hdf5_cfg.create_dataset(name='u_IC',
                                          data=jnp.frombuffer(cloudpickle.dumps(cfg.u_IC), dtype=jnp.uint8))
        tmp_set.attrs['description'] = 'Initial condition u(x,0)'
        
        hdf5_basic = file.create_group('baseline')
        tmp_set = hdf5_basic.create_dataset(name='solution', data=sol_basic, **comprex_opts)
        tmp_set.attrs['description'] = ('Solution of Burgers eq. with baseline constant '
                                        ' viscosity, k-th row is the full solution at '
                                        'time instant t_k')
        tmp_set = hdf5_basic.create_dataset(name='max_res', data=max_res_basic, **comprex_opts)
        tmp_set.attrs['description'] = 'The k-th row contains the maximum residual for the k-th time instant'
        #tmp_set = hdf5_basic.create_dataset(name='nu', data=cloudpickle.dumps(cfg.nu0_fun))
        tmp_set = hdf5_basic.create_dataset(name='nu',
                                            data=jnp.frombuffer(cloudpickle.dumps(cfg.nu0_fun), dtype=jnp.uint8))
        
        tmp_set.attrs['description'] = 'Constant viscosity function nu0(x, t, u, ux, uxx)'
        
        hdf5_exact = file.create_group('exact')
        tmp_set = hdf5_exact.create_dataset(name='solution', data=sol_var_nu, **comprex_opts)
        tmp_set.attrs['description'] = ('Solution of Burgers eq. with variable viscosity, '
                                        'k-th row is the full solution at time instant t_k')
        tmp_set = hdf5_exact.create_dataset(name='max_res', data=max_res_var_nu, **comprex_opts)
        tmp_set.attrs['description'] = 'The k-th row contains the maximum residual for the k-th time instant'
        tmp_set = hdf5_exact.create_dataset(name='nu',
                                            data=jnp.frombuffer(cloudpickle.dumps(cfg.nu), dtype=jnp.uint8))
        tmp_set.attrs['description'] = 'Exact viscosity function nu(x, t, u, ux, uxx)'
        
        hdf5_beta = file.create_group('beta')
        tmp_set = hdf5_beta.create_dataset(name='beta', data=beta_eval, **comprex_opts)
        tmp_set.attrs['description'] = 'Beta correction beta = nu_exact / nu_baseline'
        tmp_set = hdf5_beta.create_dataset(name='ux', data=ux, **comprex_opts)
        tmp_set.attrs['description'] = 'Derivative du/dx'
else:
    with h5py.File(sol_file, 'r') as file:
        t_mesh = jnp.asarray(file['mesh']['t'])
        x_mesh = jnp.asarray(file['mesh']['x'])
        max_res_basic = jnp.asarray(file['baseline']['max_res'])
        sol_basic = jnp.asarray(file['baseline']['solution'])
        max_res_var_nu = jnp.asarray(file['exact']['max_res'])
        sol_var_nu = jnp.asarray(file['exact']['solution'])
        beta_eval = jnp.asarray(file['beta']['beta'])


#%% EXTRACT "PROBES DATA"

if EXTRACT_PROBES:
    sol_probes_basic = jnp.zeros(shape=(N_t, cfg.N_probes))
    sol_probes_var_nu = jnp.zeros(shape=(N_t, cfg.N_probes))
    for j in range(N_t):
        sol_probes_basic = sol_probes_basic.at[j,:].set(jnp.interp(cfg.x_probes, x_mesh, sol_basic[j,:]))
        sol_probes_var_nu = sol_probes_var_nu.at[j,:].set(jnp.interp(cfg.x_probes, x_mesh, sol_var_nu[j,:]))
    
    print(f'Saving sampled basic solution to \n\t{probes_file.resolve()}')
    with h5py.File(probes_file, 'w') as file:
        hdf5_probes = file.create_group('probes')
        tmp_set = hdf5_probes.create_dataset(name='x', data=cfg.x_probes, **comprex_opts)
        tmp_set.attrs['description'] = 'Probe locations'
        tmp_set = hdf5_probes.create_dataset(name='t', data=t_mesh, **comprex_opts)
        tmp_set.attrs['description'] = 'Acquisition time instants'
        
        hdf5_basic = file.create_group('baseline')
        tmp_set = hdf5_basic.create_dataset(name='solution', data=sol_probes_basic, **comprex_opts)
        tmp_set.attrs['description'] = 'Sampled solution of baseline problem'
        
        hdf5_var_nu = file.create_group('exact')
        tmp_set = hdf5_var_nu.create_dataset(name='solution', data=sol_probes_var_nu, **comprex_opts)
        tmp_set.attrs['description'] = 'Sampled solution of exact problem'


#%% PLOT SOLUTION

if SOLVE:
    burgers.warning('!! CLOSE ALL FIGURES TO TERMINATE EXECUTION !!')
    
    plt.figure()
    if not STEADY:
        plt.plot(x_mesh, cfg.u_IC(x_mesh), label='$u(x,0)$')
    plt.plot(x_mesh, sol_basic[-1,:], label=r'$u$ with $\nu = \nu_0$')
    plt.plot(x_mesh, sol_var_nu[-1,:], label=r'$u$ with $\nu = \nu(x,t,u,u_x)$')
    plt.xlabel('x [m]')
    plt.ylabel('u [m/s]')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.legend()
    if not STEADY:
        plt.title(f'$t = T = {cfg.T}$ s')
    plt.show(block=False)
    
    if not STEADY:
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

if not STEADY and VIDEO:
    fig, ax = plt.subplots()
    line, = ax.plot(x_mesh, jnp.zeros_like(x_mesh))
    ax.legend(loc='upper right')
    ax.set_xlim(0, cfg.L)
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
