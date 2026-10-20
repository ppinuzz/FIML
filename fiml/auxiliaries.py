#!/usr/bin/env python3
"""
Auxiliary functions.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import numpy as np
from os import PathLike
from pathlib import Path
import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from colorama import Fore, Style, just_fix_windows_console
# call it once on Windows to fix ANSI coloured text in terminals
# (does nothing on other OS's)
just_fix_windows_console()


def warning(message):
    """
    Print yellow warning message.

    Parameters
    ----------
    message : str
        User message.

    """
    print(Fore.YELLOW + message + Style.RESET_ALL)


def check_file_or_data_input(input_obj, paran_name):
    # https://stackoverflow.com/a/76362264/17220538
    if isinstance(input_obj, PathLike):
        print(f'Reading {input_obj.resolve()}')
        input_obj = jnp.load(input_obj)
    elif isinstance(input_obj, dict) or isinstance(input_obj, np.lib.npyio.NpzFile):
        pass
    else:
        raise ValueError(f"Invalid '{paran_name}' input: either pass a Path object to a .npz file "
                         'or pass the dictionary/npz object obtained by loading')
    return input_obj


def remove_failed_runs(run_data, converged):
    N_runs = converged.size
    N_failed = N_runs - sum(converged)
    # flag failed runs and remove their data to avoid skewing the distribution
    if (~converged).any():
        warning(f'{N_failed}/{N_runs} simulation failed\n'
                'Removing failed simulation data...')
        clean_data = []
        for data in run_data:
            # ellipsis (...) is the same as ':' for all previous dimensions
            data = data[..., converged]
            clean_data.append(data)
        print(f'Number of samples modified from {N_runs} (user input) to {sum(converged)} (non-failed simulations)...')
    else:
        clean_data = run_data
    # unpack the list: specify they're part of the same 1-element return tuple
    # https://stackoverflow.com/a/74318655/17220538
    return (*clean_data, )


def check_optimiser_results(opt, SCIPY=True):
    success = True
    print(f'Cost function value at MAP point     : {opt.fun}\n'
          f'Number of optimiser iterations       : {opt.nit}\n'
          f'Number of cost function evaluations  : {opt.nfev}\n'
          f'Number of Jacobian evaluations       : {opt.njev}')
    if not opt.success:
        success = False
        warning('Failed optimisation: the optimiser could not find the '
                'maximum-a-posteiori (MAP) value by solving argmin(J(theta)).\n'
                f'Solver status: {opt.status}\n'
                f'Solver message: {opt.message}')
    if not jnp.all(jnp.isfinite(opt.x)):
        success = False
        warning('Non-finite MAP coefficients')
    if not jnp.all(jnp.isfinite(opt.jac)):
        success = False
        warning('Non-finite MAP Jacobian')
    if 'hess' in opt.keys():
        if not jnp.all(jnp.isfinite(opt.hess)):
            success = False
            warning('Non-finite MAP Hessian')
    if not success:
        raise RuntimeError('Optimisation either failed or produced invalid results')


def _is_symmetric(A):
    symmetric = jnp.allclose(A, A.T, rtol=1e-10, atol=1e-12)
    return symmetric


def _is_positive_semi_definite(A, semidef=True):
    if semidef:
        A_test = A + jnp.eye(A)*1e-14
    else:
        A_test = A
    # for REAL and SYMMETRIC matrices, Cholesky decomposition returns
    # LinAlgError if the matrix is not positive-definite
    # https://stackoverflow.com/a/44287862/17220538
    # if it's positive SEMI-definite (which is okay), Cholesky will fail
    # but the matrix is still good => add a small diagonale offset
    # https://stackoverflow.com/a/63911811/17220538
    # for JAX, this does NOT return a LinAlgError, but NaNs
    R_chol = jnp.linalg.cholesky(A_test)
    positive_semi_def = jnp.all(jnp.isfinite(R_chol))
    return positive_semi_def


def check_covariance_matrix(Cm):
    Cm = jnp.squeeze(Cm)
    if Cm.ndim > 2:
        raise ValueError('Invalid number of dimensions: covariance matrix can '
                         'only be 0D, 1D (diagonal vector) or 2D')
    # works for 0D, 1D, 2D
    finite = jnp.all(jnp.isfinite(Cm))
    if not finite:
        raise ValueError('Covariance matrix contains non-finite values')
    
    if Cm.ndim == 0 or Cm.ndim == 1:
        # matrix is only diagonal => diagonal elements are variances, not
        # covariances, therefore should all be > 0 strictly
        positive_semi_def = (Cm > 0)
        symmetric = True
    else:
        positive_semi_def = (jnp.diagonal(Cm) > 0)
    # full covariance matrix (not diagonal-only)
    if Cm.ndim == 2:
        # covariance matrix should be symmetric and semipositive-definite
        # https://www.statlect.com/fundamentals-of-probability/covariance-matrix
        symmetric = _is_symmetric(Cm)
        if not symmetric:
            raise ValueError('Covariance matrix is not symmetric')
        
        positive_semi_def = _is_positive_semi_definite(Cm, semidef=True)
        if not positive_semi_def:
            raise ValueError('Covariance matrix is not positive semi-definite')
    


def save_fig(fig, fig_name, fig_dir):
    """
    Save Matplotlib figure in PDF, PNG and SVG ensuring minimum white margins.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Handle to a Matplotlib figure to save.
    fig_name : str
        Filename of the figure once saved.
    fig_dir : str or pathlib.Path
        Path of the directory where the figure will be saved.

    """
    
    fig_dir = Path(fig_dir)
    # create the directory (if it already exists, does nothing)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(fig_dir / f'{fig_name}.pdf', bbox_inches='tight')
    fig.savefig(fig_dir / f'{fig_name}.png', bbox_inches='tight')
    fig.savefig(fig_dir / f'{fig_name}.svg', bbox_inches='tight')
        
        
@jax.jit
def sample_solution_probes(solution, x_mesh, x_probes):
    # interpolate u(x, t=t^k) from mesh to probe locations
    interpolate = lambda u_t: jnp.interp(x_probes, x_mesh, u_t)
    # apply it to all time instants
    u_interp = jax.vmap(interpolate)(solution)
    return u_interp