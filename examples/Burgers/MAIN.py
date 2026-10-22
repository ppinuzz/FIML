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
import os

from burgers import solve_Burgers
FIML_pkg = Path(r'C:\Andrea\Dottorato\FIML')
if str(FIML_pkg) not in sys.path:
    sys.path.append(str(FIML_pkg))
import fiml



#%% USER INPUT

POST_PROC = False
ANALYSIS = False
if len(sys.argv) > 1:
    # 0: filename
    # 1: flag
    # 2: flag argument
    j = 1
    while j < len(sys.argv):
        match sys.argv[j]:
            case '--input' | '-i':
                # current argument is the flag => next is the filename
                input_file = Path(os.getcwd()) / sys.argv[j+1]
                # skip filename, so that next loop will read the next flag
                j += 2
                ANALYSIS = True
                output_file = None
            case '--post' | '-p':
                # current argument is the flag => next is the filename
                output_file = Path(os.getcwd()) / sys.argv[j+1]
                # skip filename, so that next loop will read the next flag
                j += 2
                POST_PROC = False
                input_file = None
            case '--help' | '-h':
                print('Possible flags: \n'
                      '\t --input, -i: prescribe YAML input file \n'
                      '\t --post, -p: prescribe YAML postprocessing file \n'
                      '\t --help: print this help')
                quit()
            case _:
                raise ValueError('Unknown CLI input')
else:
    # interactive execution from Spyder
    ANALYSIS = True
    POST_PROC = False
    if ANALYSIS:
        input_file = Path('test_input.yaml')
        #output_file = Path('test_postproc.yaml')
        output_file = None
    if POST_PROC:
        input_file = None
        output_file = Path('Burgers_post.yaml')


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
    BATCH = False
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
    
    SEED = input_data['seed']
    RUN_PARALLEL = input_data['run_parallel']
    N_WORKERS = int(input_data['parallel_workers'])
    CHECK = input_data['check_outputs']
    VIDEO = input_data['show_videos']
    BATCH = input_data['batch_mode']


if output_file is not None:
    with open(output_file, 'r') as file:
        output_data = yaml.safe_load(file)
    data_file_out = Path(output_data['data_file'])
    
    post_dir = Path(output_data['output_dir'])


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
    if POST_PROC:
        fiml.main.postprocessing(data_file_out, post_dir)
    if ANALYSIS:
        fiml.main.analysis(exp_data, mesh, BC, nu, out_dir, N_basis_x, N_basis_t,
                       sigma2_prior, N_sampling_prior, N_sampling_post, SEED, sigma2_obs,
                       N_WORKERS, RUN_PARALLEL, solve_Burgers, CHECK, VIDEO,
                       beta_exact, BATCH, IDX_TIME_PLOT=-1)
