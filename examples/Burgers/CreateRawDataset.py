#!/usr/bin/env python3
"""
Generate raw dataset solving Burgers' equation
    u_t + uu_x = (nu(...) u_x)_x     for x in [0,L] and t in [0,T]
    u(x,0) = u0(x)
    u(0,t) = u0
    u(L,t) = uL
with different BCs and nu functions.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from pathlib import Path
import matplotlib.pyplot as plt
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

N_probes = 40

# basic setup files and common parameters
import setup_Burgers.BCDirichlet_steady_nu_u as cfg

# =============================================================================
# u0_vet = jnp.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
# uL_vet = jnp.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
# nu0_vet = jnp.array([1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1])
# =============================================================================
u0_vet = jnp.array([-2, -1, 0, 0.5, 1, 2])
uL_vet = jnp.array([-2, -1, 0, 0.5, 1, 2])
nu0_vet = jnp.array([1e-3, 1e-2, 1e-1])
beta_vet = {'monotonic_increase': lambda x, t, u, ux, uxx: (1 + 20*u**2 + 2*jnp.sin(u)),
            'downward_parabola': lambda x, t, u, ux, uxx: (1 - 50*u**2 + 55*u),
            'cubic': lambda x, t, u, ux, uxx: (2 + 10*u**3 - 10*u**2),
            'oscillating': lambda x, t, u, ux, uxx: (2 + 10*u**3 - 10*u**2)
            }
N_tot = u0_vet.size * uL_vet.size * nu0_vet.size * len(beta_vet)


#%% ------------------ DO NOT TOUCH ANYTHING BELOW THIS LINE ------------------

out_dir = Path('datasets') / Path(cfg.__file__).stem / 'raw'

out_dir.mkdir(exist_ok=True, parents=True)

# common compression options
# shuffle: allows blocks to be moved around, possible memory gain
# gzip: most portable and fairly fast
# compression_opts: for GZIP, it's the compressione level (9: maximum)
comprex_opts = {'shuffle': True, 'compression': 'gzip', 'compression_opts': 9}

x_probes = jnp.linspace(cfg.dx_in_out, cfg.L-cfg.dx_in_out, N_probes)


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


#%% SOLVE THE EQUATION

t0 = timer()
n = 1
for u0 in u0_vet:
    for uL in uL_vet:
        # if u0 = uL, the trivial solution u(x) = u0 = uL = cost. is a solution
        # of the problem => skip it
        if jnp.isclose(u0, uL) and STEADY:
            continue
        for beta_name in beta_vet:
            for nu0 in nu0_vet:
                print(f'Dataset : {n}/{N_tot}\n', 
                      f'u0      : {u0} \n'
                      f'uL      : {uL} \n'
                      f'nu0     : {nu0} \n'
                      f'beta    : {beta_name}')
                sol = jnp.zeros(shape=(N_t, N_x))
                # apply ICs and BCs
                if not STEADY:
                    sol = sol.at[0,:].set(cfg.u_IC(x_mesh))
                sol = sol.at[:,0].set(u0)
                sol = sol.at[:,-1].set(uL)
    
                nu0_fun = lambda x, t, u, ux, uxx, nu0=nu0: nu0
                beta = beta_vet[beta_name]
                nu = lambda x, t, u, ux, uxx, beta=beta, nu0_fun=nu0_fun: (
                    beta(x, t, u, ux, uxx) * nu0_fun(x, t, u, ux, uxx)
                    )
                
                start_time = timer()
                sol, max_res = burgers.solve_Burgers(sol, x_mesh, t_mesh, 
                                                            nu=nu, STEADY=STEADY)
                end_time = timer()
                time_var_nu = end_time - start_time
                print(f'Elapsed time: {time_var_nu} s ({timedelta(seconds=time_var_nu)} hours)')
    
    
                #%% EVALUATE EXACT BETA CORRECTION (for reference)
                
                ux = jnp.gradient(sol, x_mesh, axis=1)
                # jnp.gradient() is 1st order at the boundaries => manually calculate the
                # 2nd order derivatives
                dx = x_mesh[1] - x_mesh[0]
                ux = ux.at[:, 0].set(
                        (-3*sol[:, 0] + 4*sol[:, 1] - sol[:, 2]) / (2*dx)
                    )
                ux = ux.at[:, -1].set(
                        (3*sol[:, -1] - 4*sol[:, -2] + sol[:, -3]) / (2*dx)
                    )
                beta_eval = jax.vmap(lambda t, u, ux_t: beta(x_mesh, t, u, ux_t, None)
                                     )(t_mesh, sol, ux)
    
    
                #%% SAVE DATA
                
                out_dir_i = out_dir / f'u0{u0}_uL{uL}_nu0{nu0}_{beta_name}'
                sol_file = out_dir_i / 'solution.h5'
                probes_file = out_dir_i / 'probes.h5'
                out_dir_i.mkdir(parents=True, exist_ok=True)
                
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
                    tmp_set = hdf5_cfg.create_dataset(name='u0', data=u0)
                    tmp_set.attrs['description'] = 'Dirichlet boundary condition u(0,t)'
                    tmp_set = hdf5_cfg.create_dataset(name='uL', data=uL)
                    tmp_set.attrs['description'] = 'Dirichlet boundary condition u(L,t)'
                    tmp_set = hdf5_cfg.create_dataset(name='u_IC', 
                                                      data=jnp.frombuffer(cloudpickle.dumps(cfg.u_IC), dtype=jnp.uint8))
                    tmp_set.attrs['description'] = 'Initial condition u(x,0)'
                    tmp_set = hdf5_cfg.create_dataset(name='nu0',
                                                      data=jnp.frombuffer(cloudpickle.dumps(nu0_fun), dtype=jnp.uint8))
                    tmp_set.attrs['description'] = 'Constant viscosity (baseline model) nu0'
                    
                    hdf5_exact = file.create_group('exact')
                    tmp_set = hdf5_exact.create_dataset(name='solution', data=sol, **comprex_opts)
                    tmp_set.attrs['description'] = ('Solution of Burgers eq. with variable viscosity, '
                                                    'k-th row is the full solution at time instant t_k')
                    tmp_set = hdf5_exact.create_dataset(name='max_res', data=max_res, **comprex_opts)
                    tmp_set.attrs['description'] = 'The k-th row contains the maximum residual for the k-th time instant'
                    tmp_set = hdf5_exact.create_dataset(name='nu',
                                                        data=jnp.frombuffer(cloudpickle.dumps(nu), dtype=jnp.uint8))
                    tmp_set.attrs['description'] = 'Exact viscosity function nu(x, t, u, ux, uxx)'
                    
                    hdf5_beta = file.create_group('beta')
                    tmp_set = hdf5_beta.create_dataset(name='beta', data=beta_eval, **comprex_opts)
                    tmp_set.attrs['description'] = 'Beta correction beta = nu_exact / nu_baseline'
                    tmp_set = hdf5_beta.create_dataset(name='ux', data=ux, **comprex_opts)
                    tmp_set.attrs['description'] = 'Derivative du/dx'
    
    
                    #%% EXTRACT "PROBES DATA"
    
                sol_probes = jnp.zeros(shape=(N_t, N_probes))
                for j in range(N_t):
                    sol_probes = sol_probes.at[j,:].set(jnp.interp(x_probes, x_mesh, sol[j,:]))
                
                print(f'Saving sampled basic solution to \n\t{probes_file.resolve()}')
                with h5py.File(probes_file, 'w') as file:
                    hdf5_probes = file.create_group('probes')
                    tmp_set = hdf5_probes.create_dataset(name='x', data=x_probes, **comprex_opts)
                    tmp_set.attrs['description'] = 'Probe locations'
                    tmp_set = hdf5_probes.create_dataset(name='t', data=t_mesh, **comprex_opts)
                    tmp_set.attrs['description'] = 'Acquisition time instants'
                    
                    hdf5_var_nu = file.create_group('exact')
                    tmp_set = hdf5_var_nu.create_dataset(name='solution', data=sol_probes, **comprex_opts)
                    tmp_set.attrs['description'] = 'Sampled solution of exact problem'
                
                n += 1
# =============================================================================
#                 plt.figure()
#                 plt.plot(x_mesh, sol[-1,:])
#                 plt.xlabel('$x$ [-]')
#                 plt.ylabel('$u$ [-]')
#                 plt.autoscale(enable=True, axis='x', tight=True)
#                 plt.legend()
#                 plt.show(block=False)
# =============================================================================

data_time = timer() - t0
print(f'Database creation time: {data_time} s ({timedelta(seconds=data_time)} hours)')
      