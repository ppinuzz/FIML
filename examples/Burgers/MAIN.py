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


DEFAULT_INPUT = 'setup_FIML/BCDirichlet_steady_nu_u.yaml'
DEFAULT_OUTPUT = 'setup_FIML/Burgers_post.yaml'


def cli(default_input=None, default_output=None):
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
                    raise ValueError(f"Unknown CLI input '{sys.argv[j]}'")
    else:
        if default_input is not None:
            ANALYSIS = True
        if default_output is not None:
            POST_PROC = True
            
        if ANALYSIS:
            input_file = Path(default_input)
            output_file = None
        if POST_PROC:
            input_file = None
            output_file = Path(default_output)
    return ANALYSIS, POST_PROC, input_file, output_file


def main(INPUT_FILE=None, OUTPUT_FILE=None, input_data=None,
         output_data=None):
    #%% READ USER INPUT
    if (input_data, output_data) == (None, None):
        ANALYSIS, POST_PROC, input_file, output_file = cli(default_input=INPUT_FILE,
                                                           default_output=OUTPUT_FILE)
    elif input_data is not None:
        ANALYSIS = True
        POST_PROC = False
        input_file = None
    elif output_data is not None:
        POST_PROC = True
        ANALYSIS = False
        output_file = None
    if ANALYSIS:
        if input_file is not None and input_data is None:
            with open(input_file, 'r') as file:
                input_data = yaml.safe_load(file)
        sol_file = Path(input_data['files']['solution'])
        probes_file = Path(input_data['files']['data'])
        out_dir = Path(input_data['files']['output_dir'])
        
        beta_opts = fiml.beta.read_beta_options(input_data['beta'])
        
        sigma2_prior = float(input_data['variance_prior'])
        N_sampling_prior = int(input_data['num_samples_prior'])
        N_sampling_post = int(input_data['num_samples_posterior'])
        sigma2_obs = float(input_data['variance_observational'])
        
        SEED = input_data['seed']
        RUN_PARALLEL = input_data['options']['run_parallel']
        N_WORKERS = int(input_data['options']['parallel_workers'])
        CHECK = input_data['options']['check_outputs']
        VIDEO = input_data['options']['show_videos']
        BATCH = input_data['options']['batch_mode']
        STEADY = input_data['steady']
        hessian_model = input_data['hessian']
        optimiser_opts = input_data['optimisation']
    
    if POST_PROC:
        if output_file is not None and output_data is None:
            with open(output_file, 'r') as file:
                output_data = yaml.safe_load(file)
        FIML_file = Path(output_data['FIML_file'])
        probes_file = Path(output_data['data_file'])
        
        beta_exact_file = Path(output_data['exact_solution_file'])
        
        post_dir = Path(output_data['output_dir'])


    #%% REPACK DATA
    
    if ANALYSIS:
        with h5py.File(probes_file, 'r') as file:
            exp_data = {'x': jnp.asarray(file['probes']['x']),
                        't': jnp.asarray(file['probes']['t']),
                        'solution': jnp.asarray(file['exact']['solution'])
                        }
        with h5py.File(sol_file, 'r') as file:
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
        mesh = fiml.auxiliaries.generate_mesh(input_data)

    if POST_PROC:
        with h5py.File(beta_exact_file, 'r') as file:
            if 'beta' in file.keys():
                beta_exact = {'x': jnp.asarray(file['mesh']['x']),
                              't': jnp.asarray(file['mesh']['t']),
                              'beta': jnp.asarray(file['beta']['beta'])
                              }
            else:
                beta_exact = None
            sol_exact = {'x': jnp.asarray(file['mesh']['x']),
                         't': jnp.asarray(file['mesh']['t']),
                          'solution': jnp.asarray(file['exact']['solution'])
                          }
    
    #%% LAUNCH CODE
    if POST_PROC:
        fiml.main.postprocessing(FIML_file, probes_file, post_dir, beta_exact=beta_exact,
                                 sol_exact=sol_exact)
    if ANALYSIS:
        fiml.main.analysis(exp_data, mesh, BC, nu, out_dir, beta_opts,
                       sigma2_prior, N_sampling_prior, N_sampling_post, SEED, sigma2_obs,
                       N_WORKERS, RUN_PARALLEL, solve_Burgers, optimiser_opts, CHECK, VIDEO,
                       beta_exact, BATCH, IDX_TIME_PLOT=-1, STEADY=STEADY, hessian=hessian_model,
                       input_file=input_file)

if __name__ == '__main__':
    main(INPUT_FILE=DEFAULT_INPUT)
    #main(INPUT_FILE=DEFAULT_OUTPUT)