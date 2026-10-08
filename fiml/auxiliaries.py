#!/usr/bin/env python3
"""
Auxiliary functions.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import numpy as np
from scipy.interpolate import interp1d
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


def sample_solution_probes(solution, x_mesh, x_probes, saved_time=False):
    # number of time instants
    N_t = solution.shape[0]
    if saved_time:
        # time is present in 'solution' variable as 1st column
        # => +1 for time column
        sol_probes = np.zeros(shape=(N_t, x_probes.shape[0]+1))
        # copy time vector
        sol_probes[:,0] = solution[:,0].copy()
        # probe data: solution[:,idx_col_probes:]
        idx_col_probes = 1
    else:
        idx_col_probes = 0
    # interpolate along axis 1 (column axis: space) for each row of the vector
    u_interp = interp1d(x_mesh, solution[:,idx_col_probes:], axis=1, kind='linear')
    sol_probes[:,idx_col_probes:] = u_interp(x_probes)
    return sol_probes