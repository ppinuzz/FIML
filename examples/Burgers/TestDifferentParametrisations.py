#!/usr/bin/env python3
"""
Test different (mostly) Gaussian radial basis functions parametrisation

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

from pathlib import Path
import sys
import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import yaml
from copy import deepcopy
import traceback
import argparse

pkg = Path(r'C:\Andrea\Dottorato\FIML\examples\Burgers')
if str(pkg) not in sys.path:
    sys.path.append(str(pkg))
import MAIN


#%% USER INPUT

DEFAULT_INPUT = 'setup_FIML/BCDirichlet_steady_nu_u.yaml'
DEFAULT_OUTPUT = 'setup_FIML/Burgers_post.yaml'

parametrisations = ['RBF_Gauss_x', 'RBF_Gauss_u',
                    'RBF_Gauss_x_softplus', 'RBF_Gauss_u_softplus',
                    'RBF_Gauss_x_poly', 'RBF_Gauss_uu'
                    ]
# used only if this script is run on Spyder
ANALYSIS = False
POST_PROC = True


#%% PARAMETER ANALYSIS

def cli(argv=None):
    parser = argparse.ArgumentParser(
                description='Run parametric FIML analysis or postprocessing',
                # force the user to enter the exact long option's name
                # (otherwise, --a, --arg, --arg-long-name will be equivalent)
                allow_abbrev=False)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('-i', '--input')
    mode.add_argument('-p', '--post')
    
    # parse the arguments passed to the script
    args = parser.parse_args(argv)

    if args.input is not None:
        return main_analysis(args.input, parametrisations) 
    if args.post is not None:
        return main_postproc(args.post, parametrisations)
    if (args.input, args.post) == (None, None):
        if ANALYSIS:
            return main_analysis(DEFAULT_INPUT, parametrisations) 
        if POST_PROC:
            return main_postproc(DEFAULT_OUTPUT, parametrisations)
    

# move everything into a function to avoid errors due to parallel execution
# on Windows
def main_analysis(INPUT_FILE, parametrisations):

    # main input data file, which will be used as template
    with open(INPUT_FILE, 'r') as file:
        input_template = yaml.safe_load(file)
    
    outdir = Path(input_template['files']['output_dir'])
    failed = []
    
    for param in parametrisations:
        print(f'----------------- PARAMETRISATION: {param} -----------------')
        
        input_data = deepcopy(input_template)
        input_data['beta']['type'] = param
        input_data['files']['output_dir'] = str(outdir.parent/param)
    
        try:
            MAIN.main(input_data=input_data)
        except Exception:
            failed.append(param)
            print(f'ERROR OCCURRED WITH PARAMETRISATION: {param}')
            # print exception as text
            traceback.print_exc()
    
    # if failed is not []
    if failed:
        print('Failed parametrisations: ' + ', '.join(failed))


def main_postproc(INPUT_FILE, parametrisations):

    # main input data file, which will be used as template
    with open(INPUT_FILE, 'r') as file:
        output_template = yaml.safe_load(file)
    
    outdir = Path(output_template['output_dir'])
    FIML_file = Path(output_template['FIML_file'])
    data_file = Path(output_template['data_file'])
    
    for param in parametrisations:
        print(f'----------------- PARAMETRISATION: {param} -----------------')
        
        output_data = deepcopy(output_template)
        output_data['output_dir'] = str(outdir/param)
        output_data['FIML_file'] = str(FIML_file.parent.parent / param / FIML_file.name)
        output_data['data_file'] = str(data_file.parent.parent / param / data_file.name)
    
        try:
            MAIN.main(output_data=output_data)
        except Exception:
            print(f'ERROR OCCURRED WITH PARAMETRISATION: {param}')
            traceback.print_exc()


if __name__ == '__main__':
    cli()