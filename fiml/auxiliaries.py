#!/usr/bin/env python3
"""
Auxiliary functions.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import numpy as np
import jax.random as jrn
import h5py
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


def check_msg(message):
    """
    Print bright cyan check message.

    Parameters
    ----------
    message : str
        User message.

    """
    print(Fore.CYAN + Style.BRIGHT + message + Style.RESET_ALL)


def check_optimiser_results(opt, SCIPY=True):
    success = True
    check_msg(f'Cost function value at MAP point     : {opt.fun}\n'
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
        A_test = A + jnp.eye(A.shape[0])*1e-14
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
        raise ValueError('Covariance matrix can only be 0D, 1D (diagonal vector) or 2D')
    
    if not jnp.all(jnp.isfinite(Cm)):
        raise ValueError('Covariance matrix contains non-finite values')
    
    if Cm.ndim == 0 or Cm.ndim == 1:
        # matrix is only diagonal => diagonal elements are variances, not
        # covariances, therefore should all be > 0 strictly
        positive_semi_def = jnp.all(Cm > 0)
    else:
    # full covariance matrix (not diagonal-only)
        # covariance matrix should be symmetric and (semi)positive-definite
        # https://www.statlect.com/fundamentals-of-probability/covariance-matrix
        symmetric = _is_symmetric(Cm)
        if not symmetric:
            raise ValueError('Covariance matrix is not symmetric')
        
        positive_semi_def = _is_positive_semi_definite(Cm, semidef=False)
    
    if not positive_semi_def:
        raise ValueError('Covariance matrix is not positive definite')


def check_hessian(H):
    if not jnp.all(jnp.isfinite(H)):
        raise ValueError('Hessian matrix contains non-finite values')
    # Hessian matrix is always symmetric (Schwarz' theorem)
    if not _is_symmetric(H):
        raise ValueError('Hessian matrix is not symmetric')
    # if a mimimum point has been reached, the Hessian matrix should be
    # positive definite
    if not _is_positive_semi_definite(H, semidef=False):
        raise ValueError('Hessian matrix is not positive definite')


def save_fig(fig, fig_name, fig_dir):
    """
    Save Matplotlib figure in PDF, PNG and SVG ensuring minimum white margins.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Handle to a Matplotlib figure to save.
    fig_name : str or None
        Filename of the figure once saved. If None, the image won't be saved.
    fig_dir : str or pathlib.Path or None
        Path of the directory where the figure will be saved. If None, the 
        image won't be saved.

    """
    
    if fig_name is not None and fig_dir is not None:
        fig_dir = Path(fig_dir)
        # create the directory (if it already exists, does nothing)
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(fig_dir / f'{fig_name}.pdf', bbox_inches='tight')
        fig.savefig(fig_dir / f'{fig_name}.png', bbox_inches='tight')
        fig.savefig(fig_dir / f'{fig_name}.svg', bbox_inches='tight')


def sample_solution(solution, x_mesh=None, t_mesh=None, x_probes=None, t_probes=None):
    space_given = x_mesh is not None and x_probes is not None
    time_given = t_mesh is not None and t_probes is not None
    # x_probes and t_probes could be scalars => no dimension, jax.vmap will
    # return a shape error => force them to be 1D array with 1 element only
    scalar_x = False
    scalar_t = False
    if space_given:
        scalar_x = True if (jnp.ndim(x_probes)==0) else False
        x_probes = jnp.atleast_1d(x_probes)
    if time_given:
        scalar_t = True if (jnp.ndim(t_probes)==0) else False
        t_probes = jnp.atleast_1d(t_probes)
    if space_given and time_given:
        sol_space = _sample_sol_space(solution, x_mesh, x_probes)
        sol_sampled = _sample_sol_time(sol_space, t_mesh, t_probes)
    elif space_given:
        sol_sampled = _sample_sol_space(solution, x_mesh, x_probes)
    elif time_given:
        sol_sampled = _sample_sol_time(solution, t_mesh, t_probes)
    else:
        warning('Neither time nor space are fully provided (i.e. mesh values and sampling points), returning None...')
        sol_sampled = None
    # remove only the dimensions introduced for scalar coordinates
    if scalar_t:
        sol_sampled = sol_sampled[0,:]
    if scalar_x:
        sol_sampled = sol_sampled[...,0]
    return sol_sampled


def save_sampling(hdf5_file, group_name, opts, theta, sol, res, success, beta,
                  key_random=None):
    # make all data at least 1D, otherwise you won't be able to apply the
    # compression options
    theta = jnp.atleast_1d(theta)
    sol = jnp.atleast_1d(sol)
    res = jnp.atleast_1d(res)
    success = jnp.atleast_1d(success)
    beta = jnp.atleast_1d(beta)
    
    hdf5_group = hdf5_file.create_group(group_name)
    tmp_set = hdf5_group.create_dataset(name='theta', data=theta, **opts)
    tmp_set.attrs['description'] = ('Each column is one drow and theta contains the '
                                    f'{theta.shape[0]} parameters required to describe '
                                    'fully the correction beta')
    tmp_set = hdf5_group.create_dataset(name='solution', data=sol, **opts)
    tmp_set.attrs['description'] = ('The i-th row is the solution on all the 1D mesh for the i-th '
                                    'time instant')
    tmp_set = hdf5_group.create_dataset(name='max_res', data=res, **opts)
    tmp_set.attrs['description'] = ('The i-th row contains the maximum residual for the i-th time instant')
    tmp_set = hdf5_group.create_dataset(name='success', data=success, **opts)
    tmp_set.attrs['description'] = ('If False, the k-th simulation did not converge or crashed')
    tmp_set = hdf5_group.create_dataset(name='beta', data=beta, **opts)
    tmp_set.attrs['description'] = ('Correction beta(x,y) with dimensione (Nx, Nt), '
                                    'transposed ordering w.r.t. solution variable')
    if key_random is not None:
        key_random = jnp.atleast_1d(key_random)
        tmp_set = hdf5_group.create_dataset(name='key', data=key_random, **opts)
        tmp_set.attrs['description'] = ('Key used to randomly draw the parameters for the sampling')
    # write to disk without closing the file
    print(f'Writing data to \n\t{Path(hdf5_file.filename).resolve()}')
    hdf5_file.flush()


def _sample_sol_space(u, x_mesh, x_probes):
    # interpolate u(x, t=t^k) from mesh to probe locations
    interpolate = lambda u_t: jnp.interp(x_probes, x_mesh, u_t)
    if u.ndim == 1:
        u_interp = interpolate(u)
    else:
        # apply it to all time instants
        u_interp = jax.vmap(interpolate)(u)
    return u_interp


def _sample_sol_time(u, t_mesh, t_probes):
    interpolate = lambda u: jnp.interp(t_probes, t_mesh, u)
    if u.ndim == 1:
        u_interp = interpolate(u)
    else:
        # apply it to all time instants
        u_interp = jax.vmap(interpolate, in_axes=1, out_axes=1)(u)
    return u_interp


def generate_key(seed):
    if seed is None:
        # JAX doesn't support None as seed value => create a random integer 
        # using NumPy, then pass it to JAX
        # (use the max value storable in a float64)
        seed_new = np.random.default_rng(seed).integers(low=np.iinfo(np.int64).max)
    else:
        seed_new = seed
    key_random = jrn.key(seed_new)
    return key_random


def load_dataset(h5_file_ptr, group_name='/'):
    # NB: the root group name is '/'
    data = {}
    for key, dataset in h5_file_ptr[group_name].items():
        # dataset       h5py.Dataset object, vanishes once file is closed
        # dataset[:]    numpy.ndarray, read the entire array, breaks if the
        #               data stored in dataset[key] is a scalar 
        # dataset[()]   numpy.ndarray or scalar, read whatever's in the database
        data[key] = dataset[()]
    return data


def save_dict_to_hdf5(data, group_obj):
    for key in data:
        # call recursively the function till you get to a level where there
        # are no more subdictionaries
        if isinstance(data[key], dict):
            subgroup = group_obj.create_group(key)
            save_dict_to_hdf5(data[key], subgroup)
        elif data[key] is None:
            # "float" empty type
            group_obj.create_dataset(key, data=h5py.Empty('f'))
        else:
            group_obj.create_dataset(key, data=data[key])
