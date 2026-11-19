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
import yaml
from copy import deepcopy
import argparse

from burgers import solve_Burgers
# required when loading the FIML object
import burgers
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
            input_obj = fiml.config.Analysis.from_yaml(INPUT_FILE)
        else:
            input_obj = fiml.config.Analysis.from_dict(input_data)
        
        beta_opts = fiml.beta.read_beta_options(input_obj.to_dict()['beta'])
        
        
        # for reproducibility: write a copy of the input data
        run_input_file = None
        if input_obj.files.output_dir is not None:
            input_obj.files.output_dir.mkdir(parents=True, exist_ok=True)
            input_obj.to_yaml(input_obj.files.output_dir / 'input_used.yaml')
        
        
        problem, exp_data, exact_sol, cost_fun_obj = fiml.auxiliaries.read_input_data(
                                            input_obj.to_dict(), mode='analysis', 
                                            solver=solve_Burgers)
        
    
    if POST_PROC:
        if output_data is None:
            with open(OUTPUT_FILE, 'r') as file:
                output_data = yaml.safe_load(file)
        # for reproducibility: write a copy of the input data
        out_dir = Path(output_data['output_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        run_output_file = out_dir / 'post_used.yaml'
        with open(run_output_file, 'w') as file:
            yaml.safe_dump(output_data, file)
        
        problem, sol_exact, exp_data, FIML = fiml.auxiliaries.read_input_data(output_data, mode='postprocessing')

    
    #%% LAUNCH CODE
    
    if POST_PROC:
        return fiml.main.postprocessing(FIML, exp_data, problem, out_dir, sol_exact)
    if ANALYSIS:
        return fiml.main.analysis(exp_data, problem, cost_fun_obj, beta_opts,
                                  input_obj, exact_sol, input_file=run_input_file)

if __name__ == '__main__':
    #cli(['--post', str(DEFAULT_OUTPUT)])
    cli()