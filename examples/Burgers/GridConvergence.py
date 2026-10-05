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
from prettytable import PrettyTable

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
dt_vet = [5e-3, 1.25e-3, 3.125e-4, 7.8125e-5]   # [s]
dx_vet = [2.5e-3, 1.25e-3, 6.25e-4, 3.125e-4]   # [m]


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

sol_file_basic = Path('basic')
sol_file_var_nu = Path('variable_nu')

mesh_file_basic = Path('basic')
mesh_file_var_nu = Path('variable_nu')


#%% SOLVE

if SOLVE:
    for dx in dx_vet:
        for dt in dt_vet:
            # CREATE THE MESH
            print(f'--- MESH SIZE: dx = {dx} m, TIME STEP: dt = {dt} s ---')
            
            mesh_file_basic_i = out_dir / f'{mesh_file_basic.stem}_dx{dx}_dt{dt}.mesh'
            mesh_file_var_nu_i = out_dir / f'{mesh_file_var_nu.stem}_dx{dx}_dt{dt}.mesh'
            
            # N intervals => N + 1 points
            N_x = int(np.ceil((xL-x0)/dx)) + 1
            N_t = int(np.ceil(T/dt)) + 1
            x_mesh = np.linspace(x0, xL, N_x)
            t_mesh = np.linspace(0, T, N_t)
            sol_basic = np.zeros(shape=(N_t, N_x))
            sol_var_nu = np.zeros(shape=(N_t, N_x))
        
            # SOLVE THE EQUATIONS
            sol_file_basic_i = out_dir / f'{sol_file_basic.stem}_dx{dx}_dt{dt}.dat'
            sol_var_nu_i = out_dir / f'{sol_file_var_nu.stem}_dx{dx}_dt{dt}.dat'
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
                
                # VARIABLE VISCOSITY BURGERS' EQUATION
                start_time = timer()
                sol_var_nu, max_res_var_nu = burgers.solve_Burgers(sol_var_nu, x_mesh, t_mesh, nu=nu, eq_type='variable_nu')
                end_time = timer()
                time_var_nu = end_time - start_time
                print(f'Elapsed time: {time_var_nu} s ({timedelta(seconds=time_var_nu)} hours)')
                
                filename = out_dir / f'basic_dx{dx}_dt{dt}.npz'
                np.savez_compressed(filename, x_mesh=x_mesh, t_mesh=t_mesh, 
                                    max_res=max_res_basic, solution=sol_basic,
                                    duration=time_basic)
                filename = out_dir / f'variable_nu_dx{dx}_dt{dt}.npz'
                np.savez_compressed(filename, x_mesh=x_mesh, t_mesh=t_mesh, 
                                    max_res=max_res_var_nu, solution=sol_var_nu,
                                    duration=time_var_nu)
else:
    runtime_table = PrettyTable()
    runtime_table.title = 'Variable viscosity - Runtime [s]'
    runtime_table.field_names = [''] + [f'dt {dt:.3e}' for dt in dt_vet]
    
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
        row = [f'dx {dx:.3e}']
        for dt in dt_vet:
            filename = out_dir / f'basic_dx{dx}_dt{dt}.npz'
            data = np.load(filename)
            meshes_basic[dx][dt] = data['x_mesh']
            solutions_basic[dx][dt] = data['solution']
            
            filename = out_dir / f'variable_nu_dx{dx}_dt{dt}.npz'
            data = np.load(filename)
            meshes_var_nu[dx][dt] = data['x_mesh']
            solutions_var_nu[dx][dt] = data['solution']
            row.append(f"{data['duration']:.2f}")
        runtime_table.add_row(row)

    plt.figure()
    for dx, dt in zip(dx_vet, dt_vet):
        mesh = meshes_basic[dx][dt]
        sol = solutions_basic[dx][dt]
        # 0th column: time, 1st column: mas residual, 2nd+ columns: solution
        plt.plot(mesh, sol[-1,:], label=rf'$\Delta x = {dx*1e3}$ mm, $\Delta t = {dt}$ s')
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
        plt.plot(mesh, sol[-1,:], label=rf'$\Delta x = {dx*1e3}$ mm, $\Delta t = {dt}$ s')
    plt.xlabel('$x$ [m]')
    plt.ylabel('$u$ [m/s]')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.legend()
    plt.title(rf'Variable viscosity Burgers ($\nu = \nu(x,t)$), $t = {sol[-1,0]}$ s')
    # non blocking, so that it won't wait for the user to close the window
    plt.show(block=False)
    
    print(runtime_table)