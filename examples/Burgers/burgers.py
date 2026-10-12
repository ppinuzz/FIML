"""
Functions to solve Burgers' equation.

References:
    [Fornberg1988]: Fornberg - Generation of Finite Difference Formulas on 
    Arbitrarily Spaced Grids (1988)

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import numpy as np
from prettytable import PrettyTable
from scipy.linalg import solve_banded

import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


def u_x(u, dx):
    """2nd-order approximation of u_x"""
    
    # (u_i+1 - u_i-1) / 2dx, where
    #   u_i+1   is defined for i = 0, ..., N-2
    #   u_i-1   is defined for i = 1, ..., N
    # so the entire derivative is defined for the smallest subset i = 1, ..., N-2
    # if i = 1, then the derivative is
    #   (u_2 - u_0) / 2dx
    # if i = N-2, it is
    #   (u_N-1 - u_N-3) / 2dx, equivalent to (u[-1] - u[-3])/2dx
    # u[-3] is obtained using the interval indexing [0,-2) = [0,-3]
    ux_int = (u[2:] - u[:-2]) / (2*dx)
    ux_sx = (-3*u[0] + 4*u[1] - u[2]) / (2*dx)
    ux_dx = (3*u[-1] - 4*u[-2] + 1*u[-3]) / (2*dx)
    # turn scalars (dim 0, shape ()) to 1-dim array (1,) before concatenating
    dudx = jnp.concatenate((jnp.atleast_1d(ux_sx),
                             ux_int,
                             jnp.atleast_1d(ux_dx)
                            ))
    return dudx


def u_xx(u, dx):
    """2nd-order approximation of u_xx"""
    
    uxx_int = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    uxx_sx = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
    uxx_dx = (2*u[-1] - 5*u[-2] + 4*u[-3] - u[-4]) / dx**2
    # turn scalars (dim 0, shape ()) to 1-dim array (1,) before concatenating
    dudxx = jnp.concatenate((jnp.atleast_1d(uxx_sx),
                             uxx_int,
                             jnp.atleast_1d(uxx_dx)
                            ))
    return dudxx


def nu_faces(u, x, t, nu):
    """Viscosity evaluated at face nodes (e.g. i+1/2)"""
    
    dx = x[1] - x[0]
    # coordinates of face nodes
    x_f = (x[:-1] + x[1:]) / 2
    u_f = (u[:-1] + u[1:]) / 2
    # (u_i+1 - u_i) / dx is 1st order when approximating dudx_i, but 2nd order
    # when approximating dudx_i+1/2
    ux_f = (u[1:] - u[:-1]) / dx
    uxx = u_xx(u, dx)
    uxx_f = (uxx[:-1] + uxx[1:]) / 2
    nu_f = nu(x_f, t, u_f, ux_f, uxx_f)
    # make sure that the same is the same as that of the derivative term, as
    # they will be multiplied together (e.g. if nu() is a constant-value
    # function, it will return a scalar)
    nu_f = jnp.broadcast_to(nu_f, ux_f.shape)
    return nu_f


def residual(u, u_old, x, t, dt, nu):
    """
    Evaluate Burgers' equation to calculate the residual
        R(u) = du/dt + u du/dx - d(nu * du/dx)/dx
    """
    dx = x[1] - x[0]
    # 1st order backward in time
    dudt = (u - u_old) / dt
    # 2nd order centred in space @ grid nodes
    dudx = u_x(u, dx)
    nu_f = nu_faces(u, x, t, nu)
    nu_plus = nu_f[1:]
    nu_minus = nu_f[:-1]
    # nu_i+1/2 * (du/dx)_i+1/2
    nududx_plus = nu_plus * (u[2:] - u[1:-1]) / dx
    # nu_i-1/2 * (du/dx)_i-1/2
    nududx_minus = nu_minus * (u[1:-1] - u[:-2]) / dx
    d2udx2_nu = (nududx_plus - nududx_minus) / dx
    # residuals are 0 at x = 0 and x = L because BCs are satisfied exactly
    res = jnp.zeros_like(u)
    res = res.at[1:-1].set(dudt[1:-1] + u[1:-1]*dudx[1:-1] - d2udx2_nu)
    return res
    


def solve_Burgers(sol, x_mesh, t_mesh, nu, tol=1e-5, k_max=5000, WRITE_LOG=True):

    # CHECK
    # u_xx requires 4 spatial nodes: 
    #   *) u_j-1, u_j, u_j+1 for interior nodes
    #   *) u0, u1, u2, u3 for left boundary node (and -1,-2,-3,-4 for right)
    if x_mesh.size < 4:
        raise ValueError('At least 4 spatial nodes are required')
    if t_mesh.size < 2:
        raise ValueError('At least 2 time instants are required')
    

    # make sure inputs are JAX arrays
    sol = jnp.asarray(sol, dtype=jnp.float64)
    x_mesh = jnp.asarray(x_mesh, dtype=jnp.float64)
    t_mesh = jnp.asarray(t_mesh, dtype=jnp.float64)
    
    # HACK: assumes uniform time mesh
    dt = t_mesh[1] - t_mesh[0]
    N_t = t_mesh.shape[0]
    max_res = jnp.zeros(shape=t_mesh.shape)
    
    if WRITE_LOG:
        solver_table = PrettyTable()
        solver_table.title = "Burgers' equation"
        solver_table.field_names = ['t [s]', 'Max residual u [m/s2]', 'Iterations']
    
    # time t = t0 is already known from the BC => no need to solve it
    # (and residuals are 0 at that time)
    for j in range(1, N_t):
        t = t_mesh[j]
        u_old = sol[j-1,:].copy()
        # initial guess: previous time step
        u = u_old.copy()
        # assign the correct BCs, if they're time varying
        u = u.at[0].set(sol[j,0])
        u = u.at[-1].set(sol[j,-1])
        err = tol + 1
        k = 0
        while err >= tol and k < k_max:
            # 1st and last point are given by BCs => only solver internal points
            u_internal = u[1:-1].copy()
            # 1st and last residuals are 0 by def. (imposed BCs)
            res_internal = residual(u, u_old, x_mesh, t, dt, nu)[1:-1]
            # calculate J_ij = dR_i/du_j
            J = jax.jacfwd(residual)(
                            u, u_old, x_mesh, t, dt, nu)[1:-1, 1:-1]
            delta = jnp.linalg.solve(J, -res_internal)
            # update the initial guess:
            #   - u(0) and u(L) are unchanged, as they're given by the BCs and
            #   are enforced at all time instants
            #   - internal points must be updated
            u = u.at[1:-1].add(delta)
            err = jnp.max(jnp.abs(residual(u, u_old, x_mesh, t, dt, nu)[1:-1]))
            k += 1
        max_res = max_res.at[j].set(err)
        sol = sol.at[j,:].set(u)
        
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
            
            if err >= tol:
                print(f'Solver did not converge for t = {t} s, max residual: {err:.5e} m/s2')
    if WRITE_LOG:
        # once the loop has finished, print the bottom ruler
        #print(row.split('\n')[-1])     # crashes if only 1 time instant is solved
        print(solver_table.get_string().split('\n')[-1])
    return sol, max_res
