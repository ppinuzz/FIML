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


@jax.jit
def sample_solution_probes(solution, x_mesh, x_probes):
    # interpolate u(x, t=t^k) from mesh to probe locations
    interpolate = lambda u_t: jnp.interp(x_probes, x_mesh, u_t)
    # apply it to all time instants
    u_interp = jax.vmap(interpolate)(solution)
    return u_interp