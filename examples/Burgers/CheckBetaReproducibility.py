"""
Check whether the basis functions can represent accurately the exact beta.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from pathlib import Path
import sys
FIML_pkg = Path(r'C:\Andrea\Dottorato\FIML')
if str(FIML_pkg) not in sys.path:
    sys.path.append(str(FIML_pkg))
import fiml.fiml as fiml
import fiml.post as post


#%% USER INPUT

exact_beta_file = Path('data/beta.npz')

# same as the FIML input
N_basis_x = 5
N_basis_t = 7
BETA_MIN = 0.01


#%% READ DATA

beta_exact = jnp.load(exact_beta_file)
# beta is Nx x Nt mathematically => check that it's properly ordered
if beta_exact['beta'].shape != (beta_exact['x'].size, beta_exact['t'].size):
    beta = beta_exact['beta'].T
else:
    beta = beta_exact['beta']

# basis function matrices
B_x = fiml.create_basis_functions_matrix(N_basis_x, beta_exact['x'])
B_t = fiml.create_basis_functions_matrix(N_basis_t, beta_exact['t'])

post.check_parametrisation_fitness(beta, B_x, B_t, BETA_MIN=1e-2)