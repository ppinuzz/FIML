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
from colorama import Fore, Style, just_fix_windows_console
# call it once on Windows to fix ANSI coloured text in terminals
# (does nothing on other OS's)
just_fix_windows_console()


def warning(message):
    print(Fore.YELLOW + message + Style.RESET_ALL)


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


def nu_faces(u, x, t, nu, beta_t):
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
    # add correction
    beta_f = (beta_t[:-1] + beta_t[1:]) / 2
    nu_f = beta_f * nu_f
    # make sure that the same is the same as that of the derivative term, as
    # they will be multiplied together (e.g. if nu() is a constant-value
    # function, it will return a scalar)
    nu_f = jnp.broadcast_to(nu_f, ux_f.shape)
    return nu_f


def residual(u, u_old, x, t, dt, nu, beta_t):
    """
    Evaluate Burgers' equation to calculate the residual
        R(u) = du/dt + u du/dx - d(nu * du/dx)/dx
    """
    dx = x[1] - x[0]
    if dt is None:
        dudt = jnp.zeros_like(u)
    else:
        # 1st order backward in time
        dudt = (u - u_old) / dt
    # 2nd order centred in space @ grid nodes
    dudx = u_x(u, dx)
    nu_f = nu_faces(u, x, t, nu, beta_t)
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


# JIT-compile the function, including in the compiled object the function nu()
# (if nu is changed, then this function will be recompiled)
# EQUIVALENT FORM:
#   newton_step_tridiagonal = jax.jit(newton_step_tridiagonal, 
#                                       static_argnames=('nu', )
#                                   )
# => the decorator @partial acts as a "lambda function" without altering the
# syntax of the function definition
@partial(jax.jit, static_argnames=('nu', ))
def newton_step_tridiagonal(u, u_old, x_mesh, t, dt, nu, beta_t):
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
    F = lambda U: residual(u.at[1:-1].set(U), u_old, x_mesh, t, dt, nu, beta_t)[1:-1]

    # non-linear function F(U) can be expanded as F(U+dU) = F(U) + J(U)*dU + ...
    # => use the linearize() function to calculate
    #   F(U)        => the residual
    #   JdU_fun     => a function f(x) representing J(U)*x ("product-by-J" function)
    u_int = u[1:-1]
    res_int, JdU_fun = jax.linearize(F, u_int)
    # reconstruct the jacobian matrix using JACOBIAN COLOURING
    lower_diag_jac, main_diag_jac, upper_diag_jac = _jacobian_colouring_tridiag(JdU_fun, u_int.size)
    # Newton's method: J(x^n)(x_n+1 - x_n) = -F(x_n)
    # (NB: res_int is (N,), but matrix product requires (N,1))
    delta = jax.lax.linalg.tridiagonal_solve(
                lower_diag_jac, main_diag_jac, upper_diag_jac, -res_int[:,jnp.newaxis])
    # delta is returned with the same shape as -F(x_n), i.e. (N,1) => remove 1
    delta = jnp.squeeze(delta)
    # [1:-1] since 1st and last nodes are not solved for
    u_new = u.at[1:-1].add(delta)
    error = jnp.max(jnp.abs(residual(u_new, u_old, x_mesh, t, dt, nu, beta_t)))

    return u_new, error


def _beta_from_solution(beta, theta, u, t):
    """Evaluate a solution-dependent beta on one time slice (Nx,)."""
    # beta(theta, u) taken u with shape (Nt, Nx) and returns the full correction
    # field (Nx, Nt) => Newton solves one instant at a time, pass only the slice
    # at the required time instant
    # => add a fake "time dimension" to u(x, t^k), so that it's read as (1,Nx)
    # and returns beta as (Nx,1)
    beta_t = beta(theta, u=u[jnp.newaxis, :], t=t)
    return jnp.reshape(beta_t, u.shape)


def _jacobian_colouring_tridiag(JdU_fun, N):
    # reconstruct the jacobian matrix using JACOBIAN COLOURING
    # *) jax.jacfwd(F)(u_int) creates the full NxN jacobian => too slow
    # *) since J is tridiagonal if nu(x, t, u, u_x) using this discretisation, 
    # extract all the non-zero elements in J(U) by multiplying J(U) with 
    # ones-and-zeros vectors chosen on purpose
    i = jnp.arange(N)
    # 3xN matrix, where each row is a vector composed with
    #   row 1) 1st, 4th, 7th, ecc. element non-zero
    #   row 2) 2nd, 5th, 8th, ecc. element non-zero (row 1, but shifted)
    #   row 3) 3rd, 6th, 9th, ecc. element non-zero (row 2 shifted)
    #seeds = (i[None, :] % 3 == jnp.arange(3)[:, None]).astype(u.dtype)
    seeds = jnp.zeros((3, N))
    seeds = seeds.at[0, 0::3].set(1)
    seeds = seeds.at[1, 1::3].set(1)
    seeds = seeds.at[2, 2::3].set(1)
    # apply f(x) function to each column of the seeds matrix
    J_elems = jax.vmap(JdU_fun)(seeds)
    # extract jacobina diagonals and residuals vector
    lower_diag = J_elems[(i - 1)%3, i].at[0].set(0.0)
    main_diag = J_elems[i % 3, i]
    upper_diag = J_elems[(i + 1)%3, i].at[-1].set(0.0)
    return lower_diag, main_diag, upper_diag


@partial(jax.jit, static_argnames=('nu', 'beta_fun', 'GMRES_RESTART'))
def newton_step_beta_u(u, u_old, x_mesh, t, dt, nu, beta_fun, theta,
                              TOL=1e-8, GMRES_RESTART=30, MAX_ITER=10, TOL_RES=1e-6):
    """Newton step for solution-dependent beta(theta, u)"""

    # Original Newton: J(u^k, beta(x,t))(u^k+1 - u_k) = -F(u^k, beta(x,t))
    # Now: J(u^k, beta(x,t,u))(u^k+1 - u_k) = -F(u^k, beta(x,t,u))
    # where F(u) = du/dt + u du/dx - d(beta(x,t,u)*nu0*du/dx)/dx is the residual
    # => now J containes a term due to beta: DF/Du = dF/du + dF/dbeta dbeta/du
    # (D = total derivative, d = partial derivative):
    #   dF/du @ fixed beta  => the only term present also in the previous version
    #   dF/dbeta dbeta/du   => correction term
    #   *) dF/du is tridiagonale if nu = nu(x,t,u,ux) => can be constructed
    # using Jacobian colouring
    #   the correction is not => non-tridiagonal Jacobian

    u_int = u[1:-1]
    # residual take a beta at a fixed t=t^k => use a wapper to have F at the
    # correct time instant (and thus beta) for the differentiation operator
    def F(U):
        # 1st and last point are fixed by Dirichlet BCs => keep those constant
        u_full = u.at[1:-1].set(U)
        beta_t = _beta_from_solution(beta_fun, theta, u_full, t)
        return residual(u_full, u_old, x_mesh, t, dt, nu, beta_t)[1:-1]

    # Jacobian-vector product: J*u, obtained by linearising F = F0 + dF/du * du
    # (NB: F(u) is the residual function R(u)) 
    res_int, JdU_fun_int = jax.linearize(F, u_int)

    # GMRES can be preconditioned with a matrix close to J^-1, so that
    #   P^-1*J*du = -P^-1*F(u) is approximately I*du = -P^-1*F(u)
    # => use dF/dbeta as P, since it approximates J 
    # => calculate the Jacobian forcing JAX to consider beta as fixed
    beta_t_frozen = jax.lax.stop_gradient(_beta_from_solution(beta_fun, theta, u, t))
    F_frozen = lambda U: residual(u.at[1:-1].set(U), u_old, x_mesh, t, dt, nu, 
                                  beta_t_frozen)[1:-1]
    _, JdU_fun_frozen = jax.linearize(F_frozen, u_int)
    # reconstruct Jacobian matrix using jacobian colouring
    lower_diag_jac, main_diag_jac, upper_diag_jac = _jacobian_colouring_tridiag(JdU_fun_frozen, u_int.size)
    # to solve  P^-1*J*du = -P^-1*F(u), do not calculate P^-1 explicitly:
    # create a function applying the left preconditioning to a vector, returning
    # the product of P^-1*vector
    def apply_preconditioner(v):
        # RHS of the equation P^-1*v should have at least dimension (N,1)
        # => add the "1" dimension with jnp.newaxis
        y = jax.lax.linalg.tridiagonal_solve(
                lower_diag_jac, main_diag_jac, upper_diag_jac, v[:,jnp.newaxis])
        # make sure that the output vector has the same shape as the input vector
        # (e.g. (N,) rather than (N,1), if the input has shape (N,))
        y = jnp.reshape(y, v.shape)
        return y

    delta, _ = jax.scipy.sparse.linalg.gmres(JdU_fun_int, -res_int, M=apply_preconditioner,
                                             tol=TOL, atol=0.0, restart=GMRES_RESTART,
                                             maxiter=MAX_ITER)
    # check residuals of the linear system J*du = -res
    res_lin_sys = JdU_fun_int(delta) + res_int
    if jnp.linalg.norm(res_lin_sys) > TOL_RES:
        warning(f'GMRES linear system residual {jnp.linalg.norm(res_lin_sys):.5e} '
                f'exceeds linear system tolerance {TOL_RES:.5e}')
    print(jnp.linalg.norm(res_lin_sys))
    # 1st and last nodes are set by BCs
    u_new = u.at[1:-1].add(delta)
    beta_new = _beta_from_solution(beta_fun, theta, u_new, t)
    error = jnp.max(
                jnp.abs(
                    residual(u_new, u_old, x_mesh, t,dt, nu, beta_new)
                    )
                )
    
    return u_new, error


# NB: since tol and k_max are NOT static, they can be changed without having 
# to recompile the function
@partial(jax.jit, static_argnames=('nu', 'beta_fun', 'GMRES_RESTART'))
def _solve_all_steps(sol, x_mesh, t_mesh, nu, beta, tol, k_max,
                     theta=None, beta_fun=None, GMRES_TOL=1e-8,
                     GMRES_RESTART=30, GMRES_MAX_ITER=10):
    """
    Compiled Newton and time loops, viscosity must NOT depend on u_xx. 
    Beta correction can be a function of x, t, u. If beta = beta(x,t) only, a
    specific solver version is compiled (tridiagonal Jacobian, much faster than
    the GMRES solution for the u-dependent beta).
    """
    if t_mesh.size > 1:
        dt = t_mesh[1] - t_mesh[0]
        # loop through time starting from t[idx_time]
        # 1: skip 1st instant, given by BC
        idx_time = 1
    else:
        # there's only one time instant
        idx_time = 0
        dt = None

    def advance_time_step(u_old, inputs):
        idx_t, time, u0, uL = inputs
        # enforce BCs
        u = u_old.at[0].set(u0)
        u = u.at[-1].set(uL)

        # NB: tol, k_max, ecc. are accessible to the inner functions
        def continue_newton(state):
            _, err, iteration = state
            keep_going = (err >= tol) & (iteration < k_max) & jnp.isfinite(err)
            return keep_going

        def update_newton(state):
            u, _, iteration = state
            if beta_fun is None:
                # beta = beta(x,t) => no need to calculate it at runtime
                beta_t = beta[:,idx_t]
                u, err = newton_step_tridiagonal(u, u_old, x_mesh, time, dt, nu, beta_t)
            else:
                # beta = beta(..., u) => recalculated at each step
                u, err = newton_step_beta_u(u, u_old, x_mesh, time, dt, nu,
                                            beta_fun, theta,
                                            TOL=GMRES_TOL, 
                                            GMRES_RESTART=GMRES_RESTART, 
                                            MAX_ITER=GMRES_MAX_ITER)
            return u, err, iteration + 1

        # as long as continue_newton() returns True, execute update_newton()
        # both functions MUST takes in input the same variables
        # (here, u, error and iteration number)
        # equivalent to:
        #   state = (u, tol + 1, 0)
        #   while continue_newton(state):
        #       state = update_newton(state)
        # FIXME: potrebbe essere necessario specificare 
        # jnp.asarray(tol + 1.0, dtype=u.dtype) e jnp.int32(0)
        # per assicurare stessa array shape e dtype
        u, err, iteration = jax.lax.while_loop(continue_newton,
                                               update_newton,
                                               (u, tol+1, 0)
                                               )
        # return:
        #   - the CARRY, i.e. the new state (current new solution)
        #       => passed to the next invocation of advance_time_step()
        #   - the RECORDED OUTPUTS to be stored in the solution history
        #       => accumulated into arrays by scan(), useful for logging
        recorded_out = (u, err, iteration)
        return u, recorded_out

# =============================================================================
#     # equivalent to
#     #   U = sol[0,:]
#     #   history_output = []
#     #   for t_i, u0, uL, beta_t in zip(t_mesh[1:], sol[1:, 0], sol[1:, -1], beta[:,1:]):
#     #       U, output_iter = advance_time_step(U, (t_i, u0, uL, beta_t))
#     #       history_output.append(output_iter)
#     #   return U, np.vstack(history_output)
#     # NB: beta_ij = beta(x_i, t^j) => looping through time, use the transpose
#     # so that the loop selects the j-th column (looping works row-wise, not
#     # column-wise)
#     _, (u_sol, errors, iters) = jax.lax.scan(advance_time_step,
#                                              # initial carry value: u(x,0)
#                                              sol[0, :],
#                                              # thing to loop over
#                                              (t_mesh[idx_time:], sol[idx_time:,0], sol[idx_time:,-1], beta[:,idx_time:].T)
#     )
# =============================================================================
    # equivalent to
    #   U = sol[0,:]
    #   history_output = []
    #   idx_time_instants = jnp.arange(idx_time, t_mesh.size)
    #   for t_i, u0, uL, beta_t in zip(idx_time_instants, t_mesh[1:], sol[1:, 0], sol[1:, -1]):
    #       U, output_iter = advance_time_step(U, (idx_time_i, t_i, u0, uL))
    #       history_output.append(output_iter)
    #   return U, np.vstack(history_output)
    # using the time index to retrieve the correct beta when/if needed
    # (if beta = beta(u), it's not used and beta is calculated at runtime)
    _, (u_sol, errors, iters) = jax.lax.scan(advance_time_step,
                                             # initial carry value: u(x,0)
                                             sol[0, :],
                                             # thing to loop over
                                             (jnp.arange(idx_time, t_mesh.size),
                                                  t_mesh[idx_time:],
                                                  sol[idx_time:, 0],
                                                  sol[idx_time:, -1])
                                             )
    if dt is None:
        # if problem is steady, sol[0] contains the initial guess, not the 
        # analytically exact IC
        solution = u_sol
    else:
        solution = jnp.vstack((sol[0, :], u_sol))
    # 1st residual is 0, since u(x,0) is the known IC
    max_res = jnp.concatenate((jnp.zeros((idx_time, )), errors))
    return solution, max_res, iters


def solve_Burgers(sol, x_mesh, t_mesh, nu, tol=1e-5, k_max=5000, beta=None, WRITE_LOG=True,
                  STEADY=False, theta=None, GMRES_TOL=1e-8, GMRES_RESTART=30,
                  GMRES_MAX_ITER=10):
    if x_mesh.size < 4:
        raise ValueError('At least 4 spatial nodes are required')
    if STEADY and t_mesh.size != 1:
        raise ValueError('Steady problems require exactly one time instant')
    if not STEADY and t_mesh.size < 2:
        raise ValueError('Unsteady problems require at least two time instants')

    # ensure inputs are JAX arrays
    sol = jnp.asarray(sol, dtype=jnp.float64)
    x_mesh = jnp.asarray(x_mesh, dtype=jnp.float64)
    t_mesh = jnp.asarray(t_mesh, dtype=jnp.float64)

    if sol.shape != (t_mesh.size, x_mesh.size):
        raise ValueError('sol must have shape (len(t_mesh), len(x_mesh))')
    
    # beta can be 
    #   - beta(x,t) => calculated beforehand and passed as a (Nx, Nt) matrix
    #   - beta(..., u) => passed as a function => it's callable then!
    beta_depends_on_u = callable(beta)
    # if it's None (e.g. when solving the exact Burgers' equation for the
    # numerical experiment), set it to 1 everywhere
    if beta is None:
        beta_values = jnp.ones_like(sol.T)
        beta_fun = None
        theta_values = None
    # if given, it can either be a matrix or a function
    elif beta_depends_on_u:
        beta_fun = jnp.asarray(beta)
        beta_values = None
        if theta is None:
            raise ValueError('theta must be supplied when beta is callable')
        theta_values = jnp.asarray(theta)
    else:
        if beta.shape != (x_mesh.size, t_mesh.size):
            raise ValueError('Fixed beta must have shape (Nx, Nt)')
        beta_fun = None
        beta_values = jnp.asarray(beta)
        theta_values = None

    if STEADY:
        # using u = 0 everywhere as initial guess when u0 != 0 (or uL != 0) can
        # introduce a discontinuity betweeen initial guess and BCs => crash
        if WRITE_LOG:
            warning('Initialing steady solution with linear profile satisfying '
                    'boundary conditions...')
        sol = sol.at[0,:].set(jnp.linspace(sol[0,0], sol[0,-1], x_mesh.size))

    sol, max_res, iterations = _solve_all_steps(sol, x_mesh, t_mesh, nu, 
                                                beta_values, tol, k_max,
                                                theta=theta_values,
                                                beta_fun=beta_fun,
                                                GMRES_TOL=GMRES_TOL,
                                                GMRES_RESTART=GMRES_RESTART,
                                                GMRES_MAX_ITER=GMRES_MAX_ITER)

    if WRITE_LOG:
        solver_table = PrettyTable()
        solver_table.title = "Burgers' equation"
        solver_table.field_names = ['t [s]', 'Max residual u [m/s2]', 'Iterations']
        if STEADY:
            n_iter_0 = iterations[0]
        else:
            # 1st row: no iterations done for unsteady problems, as the solution
            # is given by the IC
            n_iter_0 = 0
        solver_table.add_row([f'{t_mesh[0]:.6f}', f'{max_res[0]:.5e}', f'{n_iter_0}'])
        for t, err, k in zip(t_mesh[1:], max_res[1:], iterations):
            solver_table.add_row([f'{t:.6f}', f'{err:.5e}', str(k)])
        print(solver_table)

    failed = (~jnp.isfinite(max_res)) | (max_res >= tol)
    # jax.lax.cond() replaces if (...) when the object is a JAX object in a JIT
    # compiled function
    jax.lax.cond(jnp.any(failed), 
                 # jax.debug.print() replaces print() in JIT-compiled functions
                 lambda _: jax.debug.print(
                         'WARNING: {N} time steps did not converge',
                         N=jnp.count_nonzero(failed)),
                 lambda _: None,
                 operand=None
                 )

    return sol, max_res
