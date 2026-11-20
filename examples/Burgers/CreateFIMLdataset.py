#!/usr/bin/env python3
"""
Read Burgers's raw dataset and create a FIML dataset from it.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
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
FIML_dataset = dataset_out_dir / 'FIML_dataset.h5'


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
    
    failed = []
    t0 = timer()
    # common compression options
    # shuffle: allows blocks to be moved around, possible memory gain
    # gzip: most portable and fairly fast
    # compression_opts: for GZIP, it's the compressione level (9: maximum)
    comprex_opts = {'shuffle': True, 'compression': 'gzip', 'compression_opts': 9}
    with h5py.File(FIML_dataset, 'w') as out_file:
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
            
            try:
                FIML_obj = MAIN.main(field_inv_data=input_data)
                
                out_file.create_dataset(name=f'{case_dir.name}',
                                        data=jnp.frombuffer(cloudpickle.dumps(FIML_obj), dtype=jnp.uint8),
                                        **comprex_opts)
            except Exception as e:
                print(f'FAILED RUN: {e}')
                failed.append([u0, uL, nu0, name])
        
        data_time = timer() - t0
        print(f'Database creation time: {data_time} s ({timedelta(seconds=data_time)} hours)')
        
        if failed:
            print('FAILED SIMULATIONS: u0, uL, nu0, name')
            for run in failed:
                print(f'{failed[0]:<5}, {failed[1]:<5}, {failed[2]:<5}, {failed[3]}')

if __name__ == '__main__':
    main()