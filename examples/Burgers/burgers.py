"""
Functions to solve Burgers' equation.

References:
    [Fornberg1988]: Fornberg - Generation of Finite Difference Formulas on 
    Arbitrarily Spaced Grids (1988)

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

from prettytable import PrettyTable

import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from functools import partial


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



@partial(jax.jit, static_argnames=('nu',))
def newton_step_tridiagonal(u, u_old, x_mesh, t, dt, nu):
    # eq. to solve:
    #   du/dt + u du/dx = d(nu*du/dx)/dx
    # same as
    #   F(x, t, u, ux) = 0
    # with 
    #   F(x, t, u, ux) = du/dt + u du/dx - d(nu*du/dx)/dx
    # i.e. F is the residual function
    
    # F holds at all nodes, from x0 to xL, but only the interior nodes must be
    # solved => wrap it to ignore those nodes
    # (since BCs must always be enforced, copy the input and force 1st and last
    # point to have BC values)
    F = lambda U: residual(u.at[1:-1].set(U), u_old, x_mesh, t, dt, nu)[1:-1]

    # evaluate residual and linearize it
    u_int = u[1:-1]
    res, matvec = jax.linearize(F, u_int)

    # recover the three Jacobian diagonals using three Jacobian-vector products
    N = u_int.size
    i = jnp.arange(N)
    seeds = (
        i[None, :] % 3 == jnp.arange(3)[:, None]
    ).astype(u.dtype)

    y = jax.vmap(matvec)(seeds)

    # extract diagonals and residuals vector, ignoring BC points
    lower_diag = y[(i - 1) % 3, i].at[0].set(0.0)
    main_diag = y[i % 3, i]
    upper_diag = y[(i + 1) % 3, i].at[-1].set(0.0)
    # Newton's method: J(x^n)(x_n+1 - x_n) = -F(x_n)
    delta = jax.lax.linalg.tridiagonal_solve(
        lower_diag, main_diag, upper_diag, -res[:, None]
    )[:, 0]
    # [1:-1] since 1st and last nodes are not solved for
    u_new = u.at[1:-1].add(delta)

    error = jnp.max(
                jnp.abs(residual(u_new, u_old, x_mesh, t, dt, nu)[1:-1])
                )

    return u_new, error


@partial(jax.jit, static_argnames=('nu', 'k_max'))
def _solve_all_steps(sol, x_mesh, t_mesh, nu, tol, k_max):
    """Compiled Newton and time loops. Viscosity must not depend on u_xx."""
    dt = t_mesh[1] - t_mesh[0]

    def advance_one_time_step(u_old, inputs):
        time, left_bc, right_bc = inputs
        # Set the boundary values at the NEW time instant.
        u = u_old.at[0].set(left_bc)
        u = u.at[-1].set(right_bc)

        def continue_newton(state):
            _, err, iteration = state
            return (err >= tol) & (iteration < k_max) & jnp.isfinite(err)

        def update_newton(state):
            u, _, iteration = state
            u, err = newton_step_tridiagonal(
                u, u_old, x_mesh, time, dt, nu
            )
            return u, err, iteration + 1

        u, err, iteration = jax.lax.while_loop(
            continue_newton,
            update_newton,
            (u, jnp.asarray(tol + 1.0, dtype=u.dtype), jnp.int32(0))
        )
        return u, (u, err, iteration)

    _, (history, errors, iterations) = jax.lax.scan(
        advance_one_time_step,
        sol[0, :],
        (t_mesh[1:], sol[1:, 0], sol[1:, -1])
    )

    solution = jnp.concatenate((sol[:1, :], history), axis=0)
    max_res = jnp.concatenate(
        (jnp.zeros((1,), dtype=sol.dtype), errors), axis=0
    )
    return solution, max_res, iterations


def solve_Burgers(sol, x_mesh, t_mesh, nu, tol=1e-5,
                  k_max=5000, WRITE_LOG=True):
    """Solve Burgers' equation with Dirichlet BCs on uniform grids.

    IMPORTANT: nu may depend on (x, t, u, ux), but NOT on uxx;
    the Newton matrix is otherwise generally pentadiagonal.
    """
    if x_mesh.size < 4:
        raise ValueError('At least 4 spatial nodes are required')
    if t_mesh.size < 2:
        raise ValueError('At least 2 time instants are required')

    sol = jnp.asarray(sol, dtype=jnp.float64)
    x_mesh = jnp.asarray(x_mesh, dtype=jnp.float64)
    t_mesh = jnp.asarray(t_mesh, dtype=jnp.float64)

    if sol.shape != (t_mesh.size, x_mesh.size):
        raise ValueError('sol must have shape (len(t_mesh), len(x_mesh))')

    sol, max_res, iterations = _solve_all_steps(
        sol, x_mesh, t_mesh, nu, tol, k_max
    )

    if WRITE_LOG:
        # One transfer/synchronization and one table rendering, after solving.
        import numpy as np
        time_host = np.asarray(t_mesh[1:])
        error_host = np.asarray(max_res[1:])
        iterations_host = np.asarray(iterations)

        solver_table = PrettyTable()
        solver_table.title = "Burgers' equation"
        solver_table.field_names = [
            't [s]', 'Max residual u [m/s2]', 'Iterations'
        ]
        for t, err, k in zip(time_host, error_host, iterations_host):
            solver_table.add_row([f'{t:.6f}', f'{err:.5e}', str(k)])
        print(solver_table)

        failed = (~np.isfinite(error_host)) | (error_host >= tol)
        if np.any(failed):
            print(f'Warning: {np.count_nonzero(failed)} time steps did not converge')

    return sol, max_res
