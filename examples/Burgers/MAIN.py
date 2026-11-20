#!/usr/bin/env python3
"""
Apply FIML methodology to Burgers' equation.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

from pathlib import Path
import sys
import argparse
from copy import deepcopy
import jax
jax.config.update('jax_enable_x64', True)
import yaml
# required when loading the FIML object
import burgers
from burgers import solve_Burgers
FIML_pkg = Path(r'C:\Andrea\Dottorato\FIML')
if str(FIML_pkg) not in sys.path:
    sys.path.append(str(FIML_pkg))
import fiml


# set to true if running the script from Spyder
SPYDER_RUN = True
FIELD_INVERSION_INPUT = Path('setup_FIML/BCDirichlet_steady_nu_u.yaml')
ML_INPUT = FIELD_INVERSION_INPUT
POSTPROC_INPUT = Path('setup_FIML/Burgers_post.yaml')


#%% RUN FUNCTIONS

def run_field_inversion(input_file=None, input_data=None):
    """Run field inversion from a YAML path or an in-memory dictionary."""
    
    if (input_file is None) == (input_data is None):
        raise ValueError('Provide either input_file or input_data, not both!')

    if input_file is not None:
        input_obj = fiml.config.FieldInversion.from_yaml(input_file)
    else:
        input_obj = fiml.config.FieldInversion.from_dict(input_data)

    # some functions still require dictionaries, not classes
    input_obj_dict = input_obj.to_dict()
    beta_opts = fiml.beta.read_beta_options(input_obj_dict['beta'])
    
    # for reproducibility: write a copy of the input data
    run_input_file = None
    out_dir = input_obj.files.output_dir
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_input_file = out_dir / 'FI_input.yaml'
        input_obj.to_yaml(run_input_file)

    problem, exp_data, exact_sol, cost_fun_obj = fiml.auxiliaries.read_input_data(
                                                    input_obj_dict, 
                                                    mode='field_inversion', 
                                                    solver=solve_Burgers)
    FI_obj = fiml.main.field_inversion(exp_data, problem, cost_fun_obj, beta_opts,
                                       input_obj, exact_sol, 
                                       input_file=run_input_file)

    return FI_obj


def run_machine_learning(input_file=None, input_data=None):
    """Reserved for the machine-learning workflow."""
    
    if (input_file is None) == (input_data is None):
        raise ValueError('Provide either input_file or input_data, not both!')
    
    if input_file is not None:
        input_obj = fiml.config.MachineLearning.from_yaml(input_file)
    else:
        input_obj = fiml.config.MachineLearning.from_dict(input_data)
    
    # for reproducibility: write a copy of the input data
    run_input_file = None
    out_dir = input_obj.files.output_dir
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_input_file = out_dir / 'ML_input.yaml'
        input_obj.to_yaml(run_input_file)
    
    ML_obj = fiml.main.machine_learning(input_obj)
    
    raise NotImplementedError('Machine-learning execution is in the making')


def run_postprocessing(input_file=None, input_data=None):
    """Run postprocessing from a YAML path or an in-memory dictionary."""
    
    if (input_file is None) == (input_data is None):
        raise ValueError('Provide either input_file or input_data, not both!')

    if input_file is not None:
        with open(input_file, 'r') as file:
            post_data = yaml.safe_load(file)
    else:
        post_data = deepcopy(input_data)

    out_dir = Path(post_data['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'post_used.yaml', 'w') as file:
        yaml.safe_dump(post_data, file)

    problem, sol_exact, exp_data, FIML = fiml.auxiliaries.read_input_data(
                                            post_data, mode='postprocessing')
    post_obj = fiml.main.postprocessing(FIML, exp_data, problem, out_dir, sol_exact)

    return post_obj


#%% CLI AND ENTRY POINT

def main(field_inv_file=None, ML_file=None, post_file=None,
         field_inv_data=None, ML_data=None, post_data=None):
    """Shared entry point: select exactly one file or dictionary."""
    inputs = (
        field_inv_file, ML_file, post_file,
        field_inv_data, ML_data, post_data
    )
    if sum([value is not None for value in inputs]) != 1:
        raise ValueError('Provide exactly one input file or data dictionary')

    if field_inv_file is not None or field_inv_data is not None:
        return run_field_inversion(input_file=field_inv_file, 
                                   input_data=field_inv_data)
    if ML_file is not None or ML_data is not None:
        return run_machine_learning(input_file=ML_file, input_data=ML_data)

    if post_file is not None or post_data is not None:
        return run_postprocessing(input_file=post_file, input_data=post_data)


def cli(argv=None):
    """Parse command-line flags and delegate to main()."""
    
    parser = argparse.ArgumentParser(
                prog='FIML',
                description='Run field inversion, machine learning, or postprocessing',
                # force the user to enter the exact long option's name
                # (otherwise, --a, --arg, --arg-long-name will be equivalent)
                allow_abbrev=False)
    # required=True makes it illegal to pass neither input of the tuple, 
    # one MUST be passed
    mode = parser.add_mutually_exclusive_group(required=True)
    # const=...     add a default value
    # nargs='?'     accepts zero or one value following the argument flag
    #               (when zero, use the default value!)
    # metavar=...   argument's name in help message 
    #               (here, it suggests it's a YAML file)
    mode.add_argument('-f', '--field-inversion', nargs='?',
                      const=FIELD_INVERSION_INPUT, metavar='YAML',
                      help='run field inversion')
    mode.add_argument('-m', '--machine-learning', nargs='?',
                      const=ML_INPUT, metavar='YAML',
                      help='run machine learning (not yet implemented)')
    mode.add_argument('-p', '--postprocessing', nargs='?',
                      const=POSTPROC_INPUT, metavar='YAML',
                      help='run postprocessing')
    
    args = parser.parse_args(argv)
    return main(field_inv_file=args.field_inversion, ML_file=args.machine_learning,
                post_file=args.postprocessing)


if __name__ == '__main__':
    if SPYDER_RUN:
        #result = main(field_inv_file=FIELD_INVERSION_INPUT)
        result = main(ML_file=ML_INPUT)
        # result = main(post_file=POSTPROC_INPUT)
        # result = main(field_inv_data=my_input_dict)
    else:
        result = cli()