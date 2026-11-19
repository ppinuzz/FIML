#!/usr/bin/env python3
"""
Read Burgers's raw dataset and create a FIML dataset from it.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
from pathlib import Path
import re
import yaml
from copy import deepcopy
from timeit import default_timer as timer
from datetime import timedelta
import sys
import cloudpickle
import h5py
import burgers
pkg = Path(r'C:\Andrea\Dottorato\FIML\examples\Burgers')
if str(pkg) not in sys.path:
    sys.path.append(str(pkg))
import MAIN


#%% INPUT

dataset_dir = Path('./datasets/BCDirichlet_steady_nu_u/raw')
dataset_out_dir = dataset_dir.parent / 'FIML'
setup_file = dataset_out_dir / 'setup.yaml'


#%% MAIN

def main():
    with open(setup_file, 'r') as file:
        input_template = yaml.safe_load(file)
    
    # (?P<u0>...)   named capture groupd with name 'u0'
    pattern = re.compile(
        r"^u0(?P<u0>[-+]?\d*\.?\d+)"
        r"_uL(?P<uL>[-+]?\d*\.?\d+)"
        r"_nu0(?P<nu>[-+]?\d*\.?\d+)"
        r"_(?P<name>.+)$"
    )
    
    case_dirs = []
    for case_dir in dataset_dir.iterdir():
        # iterdir() returns both files and directories
        if case_dir.is_dir():
            case_dirs.append(case_dir)
    print(f'Found {len(case_dirs)} raw case directories')
    
    t0 = timer()
    for i, case_dir in enumerate(case_dirs):
        match = pattern.fullmatch(case_dir.name)
        if match is None:
            continue
        u0 = float(match.group('u0'))
        uL = float(match.group('uL'))
        nu0 = float(match.group('nu'))
        name = match.group('name')
        print(f'Case n  : {i+1}/{len(case_dirs)} \n'
              f'u0      : {u0} \n'
              f'uL      : {uL} \n'
              f'nu0     : {nu0} \n'
              f'beta    : {name}')
        
        input_data = deepcopy(input_template)
        input_data['files']['data'] = str(case_dir.resolve() / input_data['files']['data'])
        input_data['files']['solution'] = str(case_dir.resolve() / input_data['files']['solution'])
        input_data['files']['output_dir'] = str(dataset_out_dir.resolve() / case_dir.name)
        
        FIML_obj = MAIN.main(input_data=input_data)
    
    data_time = timer() - t0
    print(f'Database creation time: {data_time} s ({timedelta(seconds=data_time)} hours)')


if __name__ == '__main__':
    main()