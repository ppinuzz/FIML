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
from prettytable import PrettyTable
from pathlib import Path

pkg = Path(r'C:\Andrea\Dottorato\FIML\examples\Burgers')
if str(pkg) not in sys.path:
    sys.path.append(str(pkg))
import MAIN


#%% USER INPUT

DEFAULT_INPUT = Path('setup_FIML/BCDirichlet_steady_nu_u.yaml')
DEFAULT_OUTPUT = Path('setup_FIML/Burgers_post.yaml')
SENSITIVITY_DIR = Path('./solution_Burgers') / f'{DEFAULT_INPUT.stem}' / 'sensitivity_test'

# =============================================================================
# parametrisations = ['RBF_Gauss_x', 'RBF_Gauss_u',
#                     'RBF_Gauss_x_softplus', 'RBF_Gauss_u_softplus',
#                     'RBF_Gauss_x_poly', 'RBF_Gauss_uu'
#                     ]
# variance_prior = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
# =============================================================================
parametrisations = ['RBF_Gauss_x', 'RBF_Gauss_u']
variance_prior = [0.05, 0.1]
# used only if this script is run on Spyder
ANALYSIS = True
POST_PROC = False


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
        return main_analysis(args.input, SENSITIVITY_DIR, parametrisations, variance_prior) 
    if args.post is not None:
        return main_postproc(args.post, SENSITIVITY_DIR, parametrisations, variance_prior)
    if (args.input, args.post) == (None, None):
        if ANALYSIS:
            return main_analysis(DEFAULT_INPUT, SENSITIVITY_DIR, parametrisations, variance_prior) 
        if POST_PROC:
            return main_postproc(DEFAULT_OUTPUT, SENSITIVITY_DIR, parametrisations, variance_prior)
    

# move everything into a function to avoid errors due to parallel execution
# on Windows
def main_analysis(INPUT_FILE, SENSITIVITY_DIR, parametrisations, variance_prior):

    SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)
    # main input data file, which will be used as template
    with open(INPUT_FILE, 'r') as file:
        input_template = yaml.safe_load(file)
    
    # make sure that no plots are shown or the code will either block (if no
    # exit.fiml file is present) or be very slow
    input_template['options']['batch_mode'] = True
    report = PrettyTable()
    report.title = 'Sensitivity study - Analysis'
    report.field_names = ['Parametrisation'] + [f'Var. prior = {var}' for var in variance_prior]
    for param in parametrisations:
        row_report = [param]
        print(f'----------------- PARAMETRISATION: {param} -----------------')
        for var_prior in variance_prior:
            print(f'PRIOR VARIANCE: {var_prior}')
            input_data = deepcopy(input_template)
            input_data['beta']['type'] = param
            input_data['files']['output_dir'] = str(SENSITIVITY_DIR/f'{param}_varPrior{var_prior}')
            input_data['variance_prior'] = var_prior
            try:
                MAIN.main(input_data=input_data)
                status = 'ok'
            except Exception:
                print(f'ERROR OCCURRED WITH PARAMETRISATION: {param}')
                # print exception as text
                traceback.print_exc()
                status = 'FAIL'
            row_report += [status]
        report.add_row(row_report)
    print(report)


def main_postproc(INPUT_FILE, SENSITIVITY_DIR, parametrisations, variance_prior):

    # main input data file, which will be used as template
    with open(INPUT_FILE, 'r') as file:
        output_template = yaml.safe_load(file)
    
    report = PrettyTable()
    report.title = 'Sensitivity study - Postprocessing'
    report.field_names = ['Parametrisation'] + [f'Var. prior = {var}' for var in variance_prior]
    FIML_file = Path(output_template['FIML_file'])
    data_file = Path(output_template['data_file'])
    for param in parametrisations:
        row_report = [param]
        print(f'----------------- PARAMETRISATION: {param} -----------------')
        for var_prior in variance_prior:
            print(f'PRIOR VARIANCE: {var_prior}')
            output_data = deepcopy(output_template)
            case_dir = SENSITIVITY_DIR / f'{param}_varPrior{var_prior}'
            output_data['output_dir'] = str(case_dir)
            output_data['FIML_file'] = str(case_dir / FIML_file.name)
            output_data['data_file'] = str(case_dir / data_file.name)
            try:
                MAIN.main(output_data=output_data)
                status = 'ok'
            except Exception:
                print(f'ERROR OCCURRED WITH PARAMETRISATION: {param}')
                traceback.print_exc()
                status = 'FAIL'
            row_report += [status]
        report.add_row(row_report)
    print(report)

if __name__ == '__main__':
    cli()