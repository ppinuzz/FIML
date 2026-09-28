#!/usr/bin/env python3
"""
Solve Burgers' equation
    u_t + uu_x = (nu(x,t) u_x)_x     for x in [0,L] and t in [0,T]
    u(x,0) = sin(pi*x/L)
    u(0,t) = 0
    u(L,t) = 0

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from prettytable import PrettyTable
import inspect
from timeit import default_timer as timer
from datetime import timedelta
import sys


#%% USER INPUT

VIDEO = False
SOLVE = False

if len(sys.argv) > 1:
    for j in range(1, len(sys.argv)):
        match sys.argv[j]:
            case '--solve':
                SOLVE = True
            case '--post':
                SOLVE = False
            case '--video':
                VIDEO = True
            case _:
                raise ValueError('Unknown CLI input')

# space-time domain size
L = 1           # [m]
T = 1           # [s]
nu0 = 1e-2      # [m2/s]

x0 = 0
xL = L

# discretisation steps
dt = 1e-3   # [s]
dx = 1e-3   # [m]

# initial and boundary conditions
u_IC = lambda x: np.sin(np.pi*x/L)
u0 = 0
uL = 0

# actual viscosity 
nu = lambda x, t: nu0 * (1 + 20*np.sin(np.pi*x/L)*(1-np.exp(-t)))
# dummy function for constant viscosity
nu0_fun = lambda x, t: nu0

# probes settings
N_probes = 10
dx_in_out = 0.05
x_probes = np.linspace(dx_in_out, L-dx_in_out, N_probes)


#%% ------------------ DO NOT TOUCH ANYTHING BELOW THIS LINE ------------------

sol_file_basic = Path('solution_basic.dat')
sol_file_var_nu = Path('solution_variable_nu.dat')

probes_file_basic = Path('probes_basic.dat')
probes_file_var_nu = Path('probes_variable_nu.dat')

mesh_file_basic = Path('basic.mesh')
mesh_file_var_nu = Path('variable_nu.mesh')


#%% MESH

# N intervals => N + 1 points
N_x = int(np.ceil((xL-x0)/dx)) + 1
N_t = int(np.ceil(T/dt)) + 1
x_mesh = np.linspace(x0, xL, N_x)
t_mesh = np.linspace(0, T, N_t)
sol_basic = np.zeros(shape=(N_t, N_x))
sol_var_nu = np.zeros(shape=(N_t, N_x))

print(f'Writing mesh to \n\t{mesh_file_basic.resolve()}')
np.savetxt(mesh_file_basic, x_mesh, delimiter=', ',
           header=('Mesh for Burgers equation with constant viscosity\n'
                   'x [m]'))
print(f'Writing mesh to \n\t{mesh_file_var_nu.resolve()}')
np.savetxt(mesh_file_var_nu, x_mesh, delimiter=', ',
           header=('Mesh for Burgers equation with variable viscosity\n'
                   'x [m]'))


#%% SOLVE THE EQUATION

def Burgers_basic(u, u_old, dx, dt, nu0):
    dudt = (u - u_old) / dt
    # (u_i+1 - u_i-1) / 2dx, where
    #   u_i+1   is defined for i = 0, ..., N-2
    #   u_i-1   is defined for i = 1, ..., N
    # so the entire derivative is defined for the smallest subset i = 1, ..., N-2
    # if i = 1, then the derivative is
    #   (u_2 - u_0) / 2dx
    # if i = N-2, it is
    #   (u_N-1 - u_N-3) / 2dx, equivalent to (u[-1] - u[-3])/2dx
    # u[-3] is obtained using the interval indexing [0,-2) = [0,-3]
    dudx = (u[2:] - u[:-2]) / (2*dx)
    d2udx2 = (u[:-2] - 2*u[1:-1] + u[2:]) / dx**2
    # residuals are 0 at x = 0 and x = L because BCs are satisfied exactly
    res = np.zeros_like(u)
    res[1:-1] = dudt[1:-1] + u[1:-1]*dudx - nu0*d2udx2
    return res


def Burgers_variable_nu(u, u_old, mesh, dt, t, nu):
    dx = mesh[1] - mesh[0]
    dudt = (u - u_old) / dt
    dudx = (u[2:] - u[:-2]) / (2*dx)
    nu_plus = nu(mesh[1:-1]+dx/2, t)
    nu_minus = nu(mesh[1:-1]-dx/2, t)
    nududx_plus = nu_plus * (u[2:] - u[1:-1]) / dx
    nududx_minus = nu_minus * (u[1:-1] - u[:-2]) / dx
    d2udx2_nu = (nududx_plus - nududx_minus) / dx
    # residuals are 0 at x = 0 and x = L because BCs are satisfied exactly
    res = np.zeros_like(u)
    res[1:-1] = dudt[1:-1] + u[1:-1]*dudx - d2udx2_nu
    return res


def solve_Burgers(sol, x_mesh, t_mesh, nu, eq_type=None, TOL=1e-5, k_max=5000):
    N_t = t_mesh.shape[0]
    N_x = x_mesh.shape[0]
    dt = t_mesh[1] - t_mesh[0]
    dx = x_mesh[1] - x_mesh[0]
    
    match eq_type:
        case 'basic':
            residual = lambda u, u_old, t: np.abs(Burgers_basic(u, u_old, dx, dt, nu0)).max()
            def calculate_u(u, u_old, dx, dt, nu, x_i=None, t_k=None):
                # Au_i^k + B + C = 0
                A = 1/dt + 0.5*(u[i+1] - u[i-1])/dx + 2*nu/dx**2
                B = - u_old[i] / dt
                C = - (u[i-1] + u[i+1]) * nu / dx**2
                u_i = - (B+C) / A
                return u_i
            title = "Basic Burgers' equation"
            if callable(nu):
                # any input will do, since nu(x,t) = nu0 = cost.
                nu = nu(None, None)
        case 'variable_nu':
            residual = lambda u, u_old, t: np.abs(Burgers_variable_nu(u, u_old, x_mesh, dt, t, nu)).max()
            def calculate_u(u, u_old, dx, dt, nu, x_i=None, t_k=None):
                nu_minus = nu(x_i-dx/2, t_k)
                nu_plus = nu(x_i+dx/2, t_k)
                A = 1/dt + 0.5*(u[i+1] - u[i-1])/dx + (nu_minus + nu_plus)/dx**2
                B = - u_old[i] / dt
                C = - (nu_plus*u[i+1] + nu_minus*u[i-1]) / dx**2
                u_i = - (B+C) / A
                return u_i
            title = "Burgers' equation with variable viscosity"
        case _:
            raise ValueError('Unknown equation')
    
    solver_table = PrettyTable()
    solver_table.title = title
    solver_table.field_names = ['t [s]', 'Max residual u [m/s]', 'Iterations']
    max_res = np.zeros(shape=(N_t, ))
    for j in range(1, N_t):
        t = t_mesh[j]
        u_old = sol[j-1,:].copy()
        u = u_old.copy()
        err = TOL + 1
        k = 0
        while err >= TOL and k < k_max:
            # 1st and last point are given by BCs
            for i in range(1, N_x-1):
                u[i] = calculate_u(u, u_old, dx, dt, nu=nu, x_i=x_mesh[i], t_k=t_mesh[j])
            err = residual(u, u_old, t)
            k += 1
        max_res[j] = err
        sol[j,:] = u.copy()
        
        solver_table.add_row([f'{t:.6f}', f'{err:.5e}', f'{k}'])
        # the 1st time, print the header too
        if j == 1:
            # remove last line from the header, to avoid double line of ------- only
            header = solver_table.get_string(header=True)
            # split the single string into its different lines (separated by '\n')
            # and glue all-but-the-last together
            header = '\n'.join(header.split('\n')[0:-1])
            print(header)
        else:
            # select the newly-added row (index j-1, since the 1st row
            # corrspoending to j = 1 has index 0) and print only the
            # numerical values (row -2 becaue the last row is the ruler)
            row = solver_table.get_string(start=j-1, border=True)
            print(row.split('\n')[-2])
        
        if err >= TOL:
            print(f'Solver did not converge for t = {t} s, max residual: {err:.5e} m/s')
    # once the loop has finished, print the bottom ruler
    print(row.split('\n')[-1])
    return sol, max_res



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
    sol_basic, max_res_basic = solve_Burgers(sol_basic, x_mesh, t_mesh, nu=nu0_fun, eq_type='basic')
    end_time = timer()
    time_basic = end_time - start_time
    print(f'Elapsed time: {time_basic} s ({timedelta(seconds=time_basic)} hours)')
    print(f'Saving solution to \n\t{sol_file_basic.resolve()}')
    # obtain a string representation of the lambda function used for u(x,0):
    #   'u_IC = lambda x: np.sin(np.pi*x/L)'
    header_IC = inspect.getsourcelines(u_IC)[0][0].strip()
    np.savetxt(sol_file_basic, 
               np.column_stack((t_mesh, max_res_basic, sol_basic)), delimiter=', ',
               header=('Solution of Burgers equation with constant viscosity\n'
                       f'x in [0,1], t in [0, {T}], nu0 = {nu0} m2/s\n'
                       f'{header_IC}\n'
                       'Each row contains the full solution for the i-th time instant\n'
                       't [s], max_res [m/s], u_j [m/s]'))
    
    # VARIABLE VISCOSITY BURGERS' EQUATION
    start_time = timer()
    sol_var_nu, max_res_var_nu = solve_Burgers(sol_var_nu, x_mesh, t_mesh, nu=nu, eq_type='variable_nu')
    end_time = timer()
    time_var_nu = end_time - start_time
    print(f'Elapsed time: {time_var_nu} s ({timedelta(seconds=time_var_nu)} hours)')
    print(f'Saving solution to \n\t{sol_file_var_nu.resolve()}')
    header_nu = inspect.getsourcelines(nu)[0][0].strip()
    np.savetxt(sol_file_var_nu, 
               np.column_stack((t_mesh, max_res_var_nu, sol_var_nu)), delimiter=', ',
               header=('Solution of Burgers equation with variable viscosity\n'
                       f'x in [0,1], t in [0, {T}], nu = nu(x,t) m2/s\n, nu0 = {nu0} m2/s\n'
                       f'{header_IC}\n'
                       f'{header_nu}\n'
                       'Each row contains the full solution for the i-th time instant\n'
                       't [s], max_res [m/s], u_j [m/s]'))
else:
    data = np.loadtxt(sol_file_basic, delimiter=',')
    t_mesh = data[:,0]
    max_res_basic = data[:,1]
    sol_basic = data[:,2:]
    data = np.loadtxt(sol_file_var_nu, delimiter=',')
    max_res_var_nu = data[:,1]
    sol_var_nu = data[:,2:]


#%% EXTRACT "PROBES DATA"

# +1 for time
sol_probes_basic = np.zeros(shape=(N_t, N_probes+1))
sol_probes_var_nu = np.zeros(shape=(N_t, N_probes+1))
for j in range(N_t):
    sol_probes_basic[j,0] = t_mesh[j]
    sol_probes_basic[j,1:] = np.interp(x_probes, x_mesh, sol_basic[j,:])
    sol_probes_var_nu[j,0] = t_mesh[j]
    sol_probes_var_nu[j,1:] = np.interp(x_probes, x_mesh, sol_var_nu[j,:])

header = ', '.join([f'u_{i} [m/s]' for i in range(1, N_probes+1)])
header_probes = ', '.join([f'{x_probes[i]}' for i in range(0, N_probes)])
print(f'Saving sampled basic solution to \n\t{probes_file_basic.resolve()}')
np.savetxt(probes_file_basic, sol_probes_basic, delimiter=', ',
           header=('Solution of Burgers equation with constant viscosity sampled along the axis\n'
                   f'x in [0,1], t in [0, {T}], nu0 = {nu0} m2/s\n'
                   f'Each row contains {N_probes} probe values\n'
                   f'Probe locations [m]: {header_probes}\n'
                   f't [s], residual [m/s], {header}'))

print(f'Saving sampled variable viscosity solution to \n\t{probes_file_var_nu.resolve()}')
np.savetxt(probes_file_var_nu, sol_probes_var_nu, delimiter=', ',
           header=('Solution of Burgers equation with variable viscosity sampled along the axis\n'
                   f'x in [0,1], t in [0, {T}], nu = nu(x,t) m2/s\n'
                   f'Each row contains {N_probes} probe values\n'
                   f'Probe locations [m]: {header_probes}\n'
                   f't [s], residual [m/s], {header}'))


#%% PLOT SOLUTION

plt.figure()
plt.plot(x_mesh, u_IC(x_mesh))
plt.xlabel('x [m]')
plt.ylabel('u [m/s]')
plt.autoscale(enable=True, axis='x', tight=True)
plt.title('$t = 0$ s')
plt.show(block=False)

plt.figure()
plt.plot(x_mesh, np.full_like(x_mesh, nu0), label=r'$\nu = \nu_0$')
plt.plot(x_mesh, nu(x_mesh,T), label=r'$\nu = \nu(x,t)$')
plt.xlabel('x [m]')
plt.ylabel(r'$\nu$ [m$^2$/s]')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend()
plt.title(rf'$\nu(x, t={T})$')
plt.show(block=False)

plt.figure()
plt.semilogy(t_mesh, max_res_basic, label=r'$\nu = \nu_0$')
plt.semilogy(t_mesh, max_res_var_nu, label=r'$\nu = \nu(x,t)$')
plt.xlabel('t [s]')
plt.ylabel(r'$\max|\mathcal{B}(u^k)|$ [m/s]')
plt.autoscale(enable=True, axis='x', tight=True)
plt.title('Maximum residual')
plt.legend()
# non blocking, so that it won't wait for the user to close the window
#plt.show(block=False)
plt.show()

if VIDEO:
    fig, ax = plt.subplots()
    line_basic, = ax.plot(x_mesh, np.zeros_like(x_mesh), label=r'$\nu = \nu_0$')
    line_var_nu, = ax.plot(x_mesh, np.zeros_like(x_mesh), label=r'$\nu = \nu(x,t)$')
    ax.legend(loc='upper right')
    ax.set_xlim(0, L)
    ax.set_ylim(-2, 2)
    # enable interactive mode
    plt.ion()
    plt.show(block=False)
    for j in range(t_mesh.shape[0]):
        # stop animation if the user closed the window
        if not plt.fignum_exists(fig.number):
            break
        line_basic.set_ydata(sol_basic[j, :])
        line_var_nu.set_ydata(sol_var_nu[j, :])
        fig.canvas.draw_idle()
        #fig.canvas.flush_events()
        #plt.pause(0.0000001)
        plt.pause(0.01)
        ax.set_title(f'$t = {t_mesh[j]:.4f}$ s')
    plt.ioff()
    # show figure only if it hasn't been closed by the user
    if plt.fignum_exists(fig.number):
        plt.show()
