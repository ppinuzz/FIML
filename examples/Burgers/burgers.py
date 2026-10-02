#!/usr/bin/env python3
"""
Functions to solve Burgers' equation.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
                        <andreapinardi319@gmail.com>
"""

import numpy as np
from prettytable import PrettyTable

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


def solve_Burgers(sol, x_mesh, t_mesh, nu, eq_type=None, TOL=1e-5, k_max=5000, WRITE_LOG=True):
    N_t = t_mesh.shape[0]
    N_x = x_mesh.shape[0]
    dt = t_mesh[1] - t_mesh[0]
    dx = x_mesh[1] - x_mesh[0]
    
    match eq_type:
        case 'basic':
            if callable(nu):
                # any input will do, since nu(x,t) = nu0 = cost.
                nu = nu(None, None)
            residual = lambda u, u_old, t: np.abs(Burgers_basic(u, u_old, dx, dt, nu)).max()
            def calculate_u(u, u_old, dx, dt, nu, x_i=None, t_k=None):
                # Au_i^k + B + C = 0
                A = 1/dt + 0.5*(u[i+1] - u[i-1])/dx + 2*nu/dx**2
                B = - u_old[i] / dt
                C = - (u[i-1] + u[i+1]) * nu / dx**2
                u_i = - (B+C) / A
                return u_i
            title = "Basic Burgers' equation"
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
    
    if WRITE_LOG:
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
        
        if WRITE_LOG:
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
    if WRITE_LOG:
        # once the loop has finished, print the bottom ruler
        print(row.split('\n')[-1])
    return sol, max_res
