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
from copy import deepcopy
import h5py
import argparse

from burgers import solve_Burgers
FIML_pkg = Path(r'C:\Andrea\Dottorato\FIML')
if str(FIML_pkg) not in sys.path:
    sys.path.append(str(FIML_pkg))
import fiml


DEFAULT_INPUT = 'setup_FIML/BCDirichlet_steady_nu_u.yaml'
DEFAULT_OUTPUT = 'setup_FIML/Burgers_post.yaml'


def cli(argv=None):
    parser = argparse.ArgumentParser(
                prog='FIML',
                description='Run FIML analysis or postprocessing',
                # force the user to enter the exact long option's name
                # (otherwise, --a, --arg, --arg-long-name will be equivalent)
                allow_abbrev=False)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('-i', '--input')
    mode.add_argument('-p', '--post')
    
    # parse the arguments passed to the script
    args = parser.parse_args(argv)

    if args.post is not None:
        return main(OUTPUT_FILE=args.post)
    else:
        if args.input is not None:
            file_input = args.input
        else:
            file_input = DEFAULT_INPUT
        return main(INPUT_FILE=file_input)    


def main(INPUT_FILE=None, OUTPUT_FILE=None, input_data=None, output_data=None):
    
    # INPUT VALIDATION
    if INPUT_FILE is not None and input_data is not None:
        raise ValueError('Provide either INPUT_FILE or input_data, not both!')
    if OUTPUT_FILE is not None and output_data is not None:
        raise ValueError('Provide either OUTPUT_FILE or output_data, not both!')
    ANALYSIS = INPUT_FILE is not None or input_data is not None
    POST_PROC = OUTPUT_FILE is not None or output_data is not None
    if ANALYSIS == POST_PROC:
        raise ValueError('Provide exactly one analysis input OR one postprocessing input')
    
    
    if ANALYSIS:
        if input_data is None:
            with open(INPUT_FILE, 'r') as file:
                input_data = yaml.safe_load(file)
        # safety: use a deep copy of the dictionary to avoid modifying data
        # of the caller script
        input_data= deepcopy(input_data)
        
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
        
        # for reproducibility: write a copy of the input data
        out_dir.mkdir(parents=True, exist_ok=True)
        run_input_file = out_dir / 'input_used.yaml'
        with open(run_input_file, 'w') as file:
            yaml.safe_dump(input_data, file)
    
    if POST_PROC:
        if output_data is None:
            with open(OUTPUT_FILE, 'r') as file:
                output_data = yaml.safe_load(file)
        
        output_data = deepcopy(output_data)
                
        FIML_file = Path(output_data['FIML_file'])
        probes_file = Path(output_data['data_file'])
        beta_exact_file = Path(output_data['exact_solution_file'])
        out_dir = Path(output_data['output_dir'])


        # for reproducibility: write a copy of the input data
        out_dir.mkdir(parents=True, exist_ok=True)
        run_output_file = out_dir / 'post_used.yaml'
        with open(run_output_file, 'w') as file:
            yaml.safe_dump(output_data, file)


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
            nu = cloudpickle.loads(jnp.asarray(file['baseline']['nu']))

    
    #%% LAUNCH CODE
    if POST_PROC:
        fiml.main.postprocessing(FIML_file, probes_file, out_dir, solve_Burgers, 
                                 nu, beta_exact=beta_exact, sol_exact=sol_exact)
    if ANALYSIS:
        fiml.main.analysis(exp_data, mesh, BC, nu, out_dir, beta_opts,
                       sigma2_prior, N_sampling_prior, N_sampling_post, SEED, sigma2_obs,
                       N_WORKERS, RUN_PARALLEL, solve_Burgers, optimiser_opts, CHECK, VIDEO,
                       beta_exact, BATCH, IDX_TIME_PLOT=-1, STEADY=STEADY, hessian=hessian_model,
                       input_file=run_input_file)

if __name__ == '__main__':
    #cli(['--post', str(DEFAULT_OUTPUT)])
    cli()