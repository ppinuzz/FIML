#!/usr/bin/env python3
"""
Auxiliary functions.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

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


@jax.jit
def sample_solution_probes(solution, x_mesh, x_probes):
    # interpolate u(x, t=t^k) from mesh to probe locations
    interpolate = lambda u_t: jnp.interp(x_probes, x_mesh, u_t)
    # apply it to all time instants
    u_interp = jax.vmap(interpolate)(solution)
    return u_interp