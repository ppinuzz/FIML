#!/usr/bin/env python3
"""
Grid convergence study on Burgers' equation.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import inspect
from timeit import default_timer as timer
from datetime import timedelta
import sys
import burgers

# =============================================================================
# turbo_pkg = Path(r'C:\Andrea\Dottorato\TurboPlotUtils')
# if str(turbo_pkg) not in sys.path:
#     sys.path.append(str(turbo_pkg))
# import turboplotutils.general as GCI
# =============================================================================


#%% USER INPUT - GCI

# discretisation steps
# 1st order in time, 2nd order in space => decrease dx by 1/2 and dt by 1/4
# to keep the same refinement [Etienne2009]
dt_vet = [5e-3, 1.25e-3, 0.3125e-3]   # [s]
dx_vet = [2.5e-3, 1.25e-3, 0.625e-3]     # [m]


#%% USER INPUT - CASE SETUP

SOLVE = False
if len(sys.argv) > 1:
    for j in range(1, len(sys.argv)):
        match sys.argv[j]:
            case '--solve':
                SOLVE = True
            case '--post':
                SOLVE = False
            case _:
                raise ValueError('Unknown CLI input')

# space-time domain size
L = 1           # [m]
T = 1           # [s]
nu0 = 1e-2      # [m2/s]

x0 = 0
xL = L

# initial and boundary conditions
u_IC = lambda x: np.sin(np.pi*x/L)
u0 = 0
uL = 0

# actual viscosity 
nu = lambda x, t: nu0 * (1 + 200*np.sin(np.pi*x/L)*(1-np.exp(-t)))
# dummy function for constant viscosity
nu0_fun = lambda x, t: nu0


#%% ------------------ DO NOT TOUCH ANYTHING BELOW THIS LINE ------------------

out_dir = Path('grid_convergence')
out_dir.mkdir(exist_ok=True)

sol_file_basic = Path('basic.dat')
sol_file_var_nu = Path('variable_nu.dat')

mesh_file_basic = Path('basic.mesh')
mesh_file_var_nu = Path('variable_nu.mesh')


#%% SOLVE

if SOLVE:
    for dx in dx_vet:
        for dt in dt_vet:
            # CREATE THE MESH
            print(f'--- MESH SIZE: dx = {dx} m, TIME STEP: dt = {dt} s ---')
            
            mesh_file_basic_i = out_dir / f'{mesh_file_basic.stem}_dx{dx}_dt{dt}{mesh_file_basic.suffix}'
            mesh_file_var_nu_i = out_dir / f'{mesh_file_var_nu.stem}_dx{dx}_dt{dt}{mesh_file_var_nu.suffix}'
            
            # N intervals => N + 1 points
            N_x = int(np.ceil((xL-x0)/dx)) + 1
            N_t = int(np.ceil(T/dt)) + 1
            x_mesh = np.linspace(x0, xL, N_x)
            t_mesh = np.linspace(0, T, N_t)
            sol_basic = np.zeros(shape=(N_t, N_x))
            sol_var_nu = np.zeros(shape=(N_t, N_x))
            
            print(f'Writing mesh to \n\t{mesh_file_basic_i.resolve()}')
            np.savetxt(mesh_file_basic_i, x_mesh, delimiter=', ',
                       header=('Mesh for Burgers equation with constant viscosity\n'
                               f'dx = {dx} m\n'
                               'x [m]'))
            print(f'Writing mesh to \n\t{mesh_file_var_nu_i.resolve()}')
            np.savetxt(mesh_file_var_nu_i, x_mesh, delimiter=', ',
                       header=('Mesh for Burgers equation with variable viscosity\n'
                               f'dx = {dx} m\n'
                               'x [m]'))
        
            # SOLVE THE EQUATIONS
            sol_file_basic_i = out_dir / f'{sol_file_basic.stem}_dx{dx}_dt{dt}{sol_file_basic.suffix}'
            sol_var_nu_i = out_dir / f'{sol_file_var_nu.stem}_dx{dx}_dt{dt}{sol_file_var_nu.suffix}'
            if SOLVE:
                # apply ICs and BCs
                sol_basic[0,:] = u_IC(x_mesh)
                sol_basic[:,0] = u0
                sol_basic[:,-1] = uL
                sol_var_nu[0,:] = u_IC(x_mesh)
                sol_var_nu[:,0] = u0
                sol_var_nu[:,-1] = uL
                
                # BASIC BURGERS' EQUATION
                start_time = timer()
                sol_basic, max_res_basic = burgers.solve_Burgers(sol_basic, x_mesh, t_mesh, nu=nu0_fun, eq_type='basic')
                end_time = timer()
                time_basic = end_time - start_time
                print(f'Elapsed time: {time_basic} s ({timedelta(seconds=time_basic)} hours)')
                print(f'Saving solution to \n\t{sol_file_basic_i.resolve()}')
                # obtain a string representation of the lambda function used for u(x,0):
                #   'u_IC = lambda x: np.sin(np.pi*x/L)'
                header_IC = inspect.getsourcelines(u_IC)[0][0].strip()
                np.savetxt(sol_file_basic_i, 
                           np.column_stack((t_mesh, max_res_basic, sol_basic)), delimiter=', ',
                           header=('Solution of Burgers equation with constant viscosity\n'
                                   f'x in [0,1], t in [0, {T}], nu0 = {nu0} m2/s\n'
                                   f'{header_IC}\n'
                                   'Each row contains the full solution for the i-th time instant\n'
                                   't [s], max_res [m/s], u_j [m/s]'))
                
                # VARIABLE VISCOSITY BURGERS' EQUATION
                start_time = timer()
                sol_var_nu, max_res_var_nu = burgers.solve_Burgers(sol_var_nu, x_mesh, t_mesh, nu=nu, eq_type='variable_nu')
                end_time = timer()
                time_var_nu = end_time - start_time
                print(f'Elapsed time: {time_var_nu} s ({timedelta(seconds=time_var_nu)} hours)')
                print(f'Saving solution to \n\t{sol_var_nu_i.resolve()}')
                header_nu = inspect.getsourcelines(nu)[0][0].strip()
                np.savetxt(sol_var_nu_i, 
                           np.column_stack((t_mesh, max_res_var_nu, sol_var_nu)), delimiter=', ',
                           header=('Solution of Burgers equation with variable viscosity\n'
                                   f'x in [0,1], t in [0, {T}], nu = nu(x,t) m2/s\n, nu0 = {nu0} m2/s\n'
                                   f'{header_IC}\n'
                                   f'{header_nu}\n'
                                   'Each row contains the full solution for the i-th time instant\n'
                                   't [s], max_res [m/s], u_j [m/s]'))
else:
    # GCI POSPROCESSING
    meshes_basic = {}
    meshes_var_nu = {}
    solutions_basic = {}
    solutions_var_nu = {}
    for dx in dx_vet:
        meshes_basic[dx] = {}
        meshes_var_nu[dx] = {}
        solutions_basic[dx] = {}
        solutions_var_nu[dx] = {}
        for dt in dt_vet:
            # READ MESHES
            mesh_file_basic_i = out_dir / f'{mesh_file_basic.stem}_dx{dx}_dt{dt}{mesh_file_basic.suffix}'
            print(f'Reading mesh file \n\t{mesh_file_basic_i.resolve()}')
            mesh = np.loadtxt(mesh_file_basic_i, delimiter=',')
            meshes_basic[dx][dt] = mesh.copy()
            
            mesh_file_var_nu_i = out_dir / f'{mesh_file_var_nu.stem}_dx{dx}_dt{dt}{mesh_file_var_nu.suffix}'
            print(f'Reading mesh file \n\t{mesh_file_var_nu_i.resolve()}')
            mesh = np.loadtxt(mesh_file_var_nu_i, delimiter=',')
            meshes_var_nu[dx][dt] = mesh.copy()
            
            # READ SOLUTIONS
            sol_file_basic_i = out_dir / f'{sol_file_basic.stem}_dx{dx}_dt{dt}{sol_file_basic.suffix}'
            print(f'Reading solution file \n\t{sol_file_basic_i.resolve()}')
            sol = np.loadtxt(sol_file_basic_i, delimiter=',')
            solutions_basic[dx][dt] = sol.copy()
            
            sol_var_nu_i = out_dir / f'{sol_file_var_nu.stem}_dx{dx}_dt{dt}{sol_file_var_nu.suffix}'
            print(f'Reading solution file \n\t{sol_var_nu_i.resolve()}')
            sol = np.loadtxt(sol_var_nu_i, delimiter=',')
            solutions_var_nu[dx][dt] = sol.copy()
    

    plt.figure()
    for dx, dt in zip(dx_vet, dt_vet):
        mesh = meshes_basic[dx][dt]
        sol = solutions_basic[dx][dt]
        # 0th column: time, 1st column: mas residual, 2nd+ columns: solution
        plt.plot(mesh, sol[-1,2:], label=rf'$\Delta x = {dx*1e3}$ mm, $\Delta t = {dt}$ s')
    plt.xlabel('$x$ [m]')
    plt.ylabel('$u$ [m/s]')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.legend()
    plt.title(rf'Basic Burgers ($\nu = \nu_0$), $t = {sol[-1,0]}$ s')
    # non blocking, so that it won't wait for the user to close the window
    plt.show(block=False)
    
    plt.figure()
    for dx, dt in zip(dx_vet, dt_vet):
        mesh = meshes_var_nu[dx][dt]
        sol = solutions_var_nu[dx][dt]
        # 0th column: time, 1st column: mas residual, 2nd+ columns: solution
        plt.plot(mesh, sol[-1,2:], label=rf'$\Delta x = {dx*1e3}$ mm, $\Delta t = {dt}$ s')
    plt.xlabel('$x$ [m]')
    plt.ylabel('$u$ [m/s]')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.legend()
    plt.title(rf'Variable viscosity Burgers ($\nu = \nu(x,t)$), $t = {sol[-1,0]}$ s')
    # non blocking, so that it won't wait for the user to close the window
    plt.show(block=False)