#!/usr/bin/env python3
"""
Apply FIML methodology to Burgers' equation.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

from pathlib import Path
import sys
import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import cloudpickle
import yaml

from burgers import solve_Burgers
FIML_pkg = Path(r'C:\Andrea\Dottorato\FIML')
if str(FIML_pkg) not in sys.path:
    sys.path.append(str(FIML_pkg))
import fiml



#%% USER INPUT

input_file = Path('test.yaml')

if input_file is None:
    exact_sol_file = Path('solution/variable_nu.npz')
    cfg_file = Path('solution/settings.npz')
    exp_data_file = Path('probes/variable_nu.npz')
    beta_exact_file = Path('solution/beta.npz')
    
    out_dir = Path('samples')
    
    N_basis_x = 5
    N_basis_t = 7
    sigma2_prior = 0.4
    N_sampling_prior = 20
    N_sampling_post = 20
    sigma2_obs = 1e-2
    
    SEED = 123
    RUN_PARALLEL = True
    N_WORKERS = 4
    CHECK = True
    VIDEO = False
else:
    with open(input_file, 'r') as file:
        input_data = yaml.safe_load(file)
    exact_sol_file = Path(input_data['exact_sol_file'])
    cfg_file = Path(input_data['problem_cfg_file'])
    exp_data_file = Path(input_data['data_file'])
    beta_exact_file = Path(input_data['exact_beta_file'])
    
    out_dir = Path(input_data['output_dir'])
    
    N_basis_x = int(input_data['num_basis_functions_x'])
    N_basis_t = int(input_data['num_basis_functions_t'])
    sigma2_prior = float(input_data['variance_prior'])
    N_sampling_prior = int(input_data['num_samples_prior'])
    N_sampling_post = int(input_data['num_samples_posterior'])
    sigma2_obs = float(input_data['variance_observational'])
    
    SEED = int(input_data['seed'])
    RUN_PARALLEL = bool(input_data['run_parallel'])
    N_WORKERS = int(input_data['parallel_workers'])
    CHECK = bool(input_data['check_outputs'])
    VIDEO = bool(input_data['show_videos'])



#%% ------------------ DO NOT TOUCH ANYTHING BELOW THIS LINE ------------------

exp_data = jnp.load(exp_data_file)
beta_exact = jnp.load(beta_exact_file)
cfg = jnp.load(cfg_file)

nu = cloudpickle.loads(cfg['nu'])
mesh = {'x': cfg['x'], 
        't': cfg['t']
        }

BC = {'u0': cfg['u0'], 
      'uL': cfg['uL'], 
      'u_IC': cloudpickle.loads(cfg['u_IC'])
      }


if __name__ == '__main__':
    fiml.main.main(exp_data, mesh, BC, nu, out_dir, N_basis_x, N_basis_t,
                   sigma2_prior, N_sampling_prior, N_sampling_post, SEED, sigma2_obs,
                   N_WORKERS, RUN_PARALLEL, solve_Burgers, CHECK=False, VIDEO=False,
                   beta_exact=None, IDX_TIME_PLOT=-1)
