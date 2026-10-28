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
import h5py

from burgers import solve_Burgers
FIML_pkg = Path(r'C:\Andrea\Dottorato\FIML')
if str(FIML_pkg) not in sys.path:
    sys.path.append(str(FIML_pkg))
import fiml


#%% USER INPUT

if len(sys.argv) > 1:
    ANALYSIS = False
    POST_PROC = False
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
                POST_PROC = True
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
        input_file = Path('setup_FIML/BCDirichlet_steady_nu_u.yaml')
        output_file = None
    if POST_PROC:
        input_file = None
        output_file = Path('Burgers_post.yaml')


#%% UNPACK USER INPUT

if ANALYSIS:
    with open(input_file, 'r') as file:
        input_data = yaml.safe_load(file)
    sol_file = Path(input_data['sol_file'])
    probes_file = Path(input_data['data_file'])
    out_dir = Path(input_data['output_dir'])
    
    beta_opts = fiml.beta.read_beta_options(input_data)
    
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
    STEADY = input_data['steady']

if POST_PROC:
    with open(output_file, 'r') as file:
        output_data = yaml.safe_load(file)
    sol_file = Path(output_data['solution_file'])
    
    post_dir = Path(output_data['output_dir'])
    
    exp_data_file = Path(output_data['data_file'])
    beta_exact_file = Path(output_data['exact_beta_file'])
    cfg_file = Path(output_data['problem_cfg_file'])


#%% REPACK DATA

if ANALYSIS:
    with h5py.File(probes_file, 'r') as file:
        exp_data = {'x': jnp.asarray(file['probes']['x']),
                    't': jnp.asarray(file['probes']['t']),
                    'solution': jnp.asarray(file['exact']['solution'])
                    }
    with h5py.File(sol_file, 'r') as file:
        mesh = {'x': jnp.asarray(file['mesh']['x']),
                't': jnp.asarray(file['mesh']['t'])
                }
        BC = {'u0': jnp.asarray(file['settings']['u0']), 
              'uL': jnp.asarray(file['settings']['uL']), 
              # HDF5 dataset object -> NumPy array -> lambda function
              'u_IC': cloudpickle.loads(jnp.asarray(file['settings']['u_IC']))
              }
        nu = cloudpickle.loads(jnp.asarray(file['baseline']['nu']))
        if 'beta' in file.keys():
            beta_exact = {'x': jnp.asarray(file['mesh']['x']),
                          't': jnp.asarray(file['mesh']['t']),
                          'beta': jnp.asarray(file['beta']['beta'])
                          }
        else:
            beta_exact = None


if __name__ == '__main__':
    if POST_PROC:
        fiml.main.postprocessing(sol_file, exp_data_file, post_dir, beta_exact=beta_exact)
    if ANALYSIS:
        fiml.main.analysis(exp_data, mesh, BC, nu, out_dir, beta_opts,
                       sigma2_prior, N_sampling_prior, N_sampling_post, SEED, sigma2_obs,
                       N_WORKERS, RUN_PARALLEL, solve_Burgers, CHECK, VIDEO,
                       beta_exact, BATCH, IDX_TIME_PLOT=-1, STEADY=STEADY)
