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

pkg = Path(r'C:\Andrea\Dottorato\FIML\examples\Burgers')
if str(pkg) not in sys.path:
    sys.path.append(str(pkg))
import MAIN


#%% USER INPUT

DEFAULT_INPUT = 'setup_FIML/BCDirichlet_steady_nu_u.yaml'
DEFAULT_OUTPUT = 'setup_FIML/Burgers_post.yaml'

parametrisations = ['RBF_Gauss_x_poly', 
                    'RBF_Gauss_uu', 'RBF_Gauss_x_softplus', 'RBF_Gauss_u_softplus',
                    'RBF_Gauss_x', 'RBF_Gauss_u'
                    ]


#%% PARAMETER ANALYSIS

# move everything into a function to avoid errors due to parallel execution
# on Windows
def main_analysis(INPUT_FILE, parametrisations):

    # main input data file, which will be used as template
    with open(INPUT_FILE, 'r') as file:
        input_data = yaml.safe_load(file)
    
    for param in parametrisations:
        print(f'----------------- PARAMETRISATION: {param} -----------------')
        input_data['beta']['type'] = param
        outdir = Path(input_data['files']['output_dir'])
        input_data['files']['output_dir'] = str(outdir.parent/param)
    
        try:
            MAIN.main(input_data=input_data)
        except:
            print(f'ERROR OCCURRED WITH PARAMETRISATION: {param}')


def main_postproc(INPUT_FILE, parametrisations):

    # main input data file, which will be used as template
    with open(INPUT_FILE, 'r') as file:
        output_data = yaml.safe_load(file)
    
    for param in parametrisations:
        print(f'----------------- PARAMETRISATION: {param} -----------------')
        outdir = Path(output_data['output_dir'])
        output_data['output_dir'] = str(outdir/param)
        FIML_file = Path(output_data['FIML_file'])
        output_data['FIML_file'] = str(FIML_file.parent.parent / param / FIML_file.name)
        data_file = Path(output_data['data_file'])
        output_data['data_file'] = str(data_file.parent.parent / param / data_file.name)
    
        try:
            MAIN.main(output_data=output_data)
        except:
            print(f'ERROR OCCURRED WITH PARAMETRISATION: {param}')


if __name__ == '__main__':
    main_analysis(DEFAULT_INPUT, parametrisations)
    #main_postproc(DEFAULT_OUTPUT, parametrisations)