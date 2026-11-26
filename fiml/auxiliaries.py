#!/usr/bin/env python3
"""
Auxiliary functions.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import numpy as np
import jax.random as jrn
import h5py
from pathlib import Path
import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from colorama import Fore, Style, just_fix_windows_console
# call it once on Windows to fix ANSI coloured text in terminals
# (does nothing on other OS's)
just_fix_windows_console()
from prettytable import PrettyTable
import cloudpickle
import yaml
import re
import random
import fiml.objects as objs
from types import SimpleNamespace


def warning(message):
    """
    Print yellow warning message.

    Parameters
    ----------
    message : str
        User message.

    """
    print(Fore.YELLOW + message + Style.RESET_ALL)


def check_msg(message):
    """
    Print bright cyan check message.

    Parameters
    ----------
    message : str
        User message.

    """
    print(Fore.CYAN + Style.BRIGHT + message + Style.RESET_ALL)


def check_optimiser_results(opt, SCIPY=True):
    success = True
    check_msg(f'Cost function value at MAP point     : {opt.fun}\n'
              f'Number of optimiser iterations       : {opt.nit}\n'
              f'Number of cost function evaluations  : {opt.nfev}\n'
              f'Number of Jacobian evaluations       : {opt.njev}')
    if not opt.success:
        success = False
        warning('Failed optimisation: the optimiser could not find the '
                'maximum-a-posteiori (MAP) value by solving argmin(J(theta)).\n'
                f'Solver status: {opt.status}\n'
                f'Solver message: {opt.message}')
    if not jnp.all(jnp.isfinite(opt.x)):
        success = False
        warning('Non-finite MAP coefficients')
    if not jnp.all(jnp.isfinite(opt.jac)):
        success = False
        warning('Non-finite MAP Jacobian')
    if 'hess' in opt.keys():
        if not jnp.all(jnp.isfinite(opt.hess)):
            success = False
            warning('Non-finite MAP Hessian')
    if not success:
        raise RuntimeError('Optimisation either failed or produced invalid results')    


def _is_symmetric(A):
    symmetric = jnp.allclose(A, A.T, rtol=1e-10, atol=1e-12)
    return symmetric


def _is_positive_semi_definite(A, semidef=True):
    if semidef:
        A_test = A + jnp.eye(A.shape[0])*1e-14
    else:
        A_test = A
    # for REAL and SYMMETRIC matrices, Cholesky decomposition returns
    # LinAlgError if the matrix is not positive-definite
    # https://stackoverflow.com/a/44287862/17220538
    # if it's positive SEMI-definite (which is okay), Cholesky will fail
    # but the matrix is still good => add a small diagonale offset
    # https://stackoverflow.com/a/63911811/17220538
    # for JAX, this does NOT return a LinAlgError, but NaNs
    R_chol = jnp.linalg.cholesky(A_test)
    positive_semi_def = jnp.all(jnp.isfinite(R_chol))
    return positive_semi_def


def check_covariance_matrix(Cm):
    Cm = jnp.squeeze(Cm)
    if Cm.ndim > 2:
        raise ValueError('Covariance matrix can only be 0D, 1D (diagonal vector) or 2D')
    
    if not jnp.all(jnp.isfinite(Cm)):
        raise ValueError('Covariance matrix contains non-finite values')
    
    if Cm.ndim == 0 or Cm.ndim == 1:
        # matrix is only diagonal => diagonal elements are variances, not
        # covariances, therefore should all be > 0 strictly
        positive_semi_def = jnp.all(Cm > 0)
    else:
    # full covariance matrix (not diagonal-only)
        # covariance matrix should be symmetric and (semi)positive-definite
        # https://www.statlect.com/fundamentals-of-probability/covariance-matrix
        symmetric = _is_symmetric(Cm)
        if not symmetric:
            raise ValueError('Covariance matrix is not symmetric')
        
        positive_semi_def = _is_positive_semi_definite(Cm, semidef=False)
    
    if not positive_semi_def:
        raise ValueError('Covariance matrix is not positive definite')


def check_hessian(H):
    if not jnp.all(jnp.isfinite(H)):
        raise ValueError('Hessian matrix contains non-finite values')
    # Hessian matrix is always symmetric (Schwarz' theorem)
    if not _is_symmetric(H):
        raise ValueError('Hessian matrix is not symmetric')
    # if a mimimum point has been reached, the Hessian matrix should be
    # positive definite
    if not _is_positive_semi_definite(H, semidef=False):
        raise ValueError('Hessian matrix is not positive definite')


def save_fig(fig, fig_name, fig_dir):
    """
    Save Matplotlib figure in PDF, PNG and SVG ensuring minimum white margins.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Handle to a Matplotlib figure to save.
    fig_name : str or None
        Filename of the figure once saved. If None, the image won't be saved.
    fig_dir : str or pathlib.Path or None
        Path of the directory where the figure will be saved. If None, the 
        image won't be saved.

    """
    
    if fig_name is not None and fig_dir is not None:
        fig_dir = Path(fig_dir)
        # create the directory (if it already exists, does nothing)
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(fig_dir / f'{fig_name}.pdf', bbox_inches='tight')
        fig.savefig(fig_dir / f'{fig_name}.png', bbox_inches='tight')
        fig.savefig(fig_dir / f'{fig_name}.svg', bbox_inches='tight')


def sample_solution(solution, x_mesh=None, t_mesh=None, x_probes=None, t_probes=None):
    space_given = x_mesh is not None and x_probes is not None
    time_given = t_mesh is not None and t_probes is not None
    # x_probes and t_probes could be scalars => no dimension, jax.vmap will
    # return a shape error => force them to be 1D array with 1 element only
    scalar_x = False
    scalar_t = False
    if space_given:
        scalar_x = True if (jnp.ndim(x_probes)==0) else False
        x_probes = jnp.atleast_1d(x_probes)
    if time_given:
        scalar_t = True if (jnp.ndim(t_probes)==0) else False
        t_probes = jnp.atleast_1d(t_probes)
    if space_given and time_given:
        sol_space = _sample_sol_space(solution, x_mesh, x_probes)
        sol_sampled = _sample_sol_time(sol_space, t_mesh, t_probes)
    elif space_given:
        sol_sampled = _sample_sol_space(solution, x_mesh, x_probes)
    elif time_given:
        sol_sampled = _sample_sol_time(solution, t_mesh, t_probes)
    else:
        warning('Neither time nor space are fully provided (i.e. mesh values and sampling points), returning None...')
        sol_sampled = None
    # remove only the dimensions introduced for scalar coordinates
    if scalar_t:
        sol_sampled = sol_sampled[0,:]
    if scalar_x:
        sol_sampled = sol_sampled[...,0]
    return sol_sampled


def save_sampling(hdf5_file, group_name, opts, data_obj, save_failed_runs=False):
    
    FIELDS_SAVE = {'theta': ('Each column is one drow and theta contains the '
                             f'{data_obj.theta.shape[0]} parameters required to '
                             'describe fully the correction beta'), 
                   'solution': ('The i-th row is the solution on all the 1D mesh '
                                'for the i-th time instant'),
                   'max_res': ('The i-th row contains the maximum residual for '
                               'the i-th time instant'),
                   'success': ('If False, the k-th simulation did not converge '
                               'or crashed'), 
                   'beta': ('Correction beta(x,y) with dimensione (Nx, Nt), '
                            'transposed ordering w.r.t. solution variable'),
                   'key_random': ('Key used to randomly draw the parameters '
                                  'for the sampling'),
                   # these 3 are available only with FIML.MAP, ignored otherwise
                   'covariance': 'Covariance matrix calculated as H(theta_MAP)^-1',
                   'jacobian': 'Jacobian matrix of the cost function J(theta_MAP)',
                   'hessian': 'Hessian matrix of the cost function J(theta_MAP)'
                   }
    
    print(f"Saving the following fields in HDF5 group '{group_name}': {list(FIELDS_SAVE.keys())}")
    if save_failed_runs:
        print('Saving failed runs (non-converged solution or Inf/NaN solution) '
              'as well as filtered runs, with only converged solutions')
    
    hdf5_group = hdf5_file.create_group(group_name)
    for field in FIELDS_SAVE:
        # None as default value
        data = getattr(data_obj, field, None)
        if data is None:
            warning(f"Property '{field}' not present in data object, skipping it...")
            continue
        # make data at least 1D, otherwise you won't be able to apply the
        # compression options
        data = jnp.atleast_1d(data)
    
        tmp_set = hdf5_group.create_dataset(name=field, data=data, **opts)
        tmp_set.attrs['description'] = FIELDS_SAVE[field]
        
        if save_failed_runs:
            data = getattr(data_obj, f'{field}_orig', None)
            if data is None:
                warning(f"Property '{field}_orig' not present in data object, skipping it...")
                continue
            tmp_set = hdf5_group.create_dataset(name=f'{field}_orig', data=data, **opts)
            tmp_set.attrs['description'] = (f'Same as {FIELDS_SAVE[field]}, but '
                                            'original data, including failed or '
                                            'non-converged runs')

    # write to disk without closing the file
    print(f'Writing data to \n\t{Path(hdf5_file.filename).resolve()}')
    hdf5_file.flush()



def _sample_sol_space(u, x_mesh, x_probes):
    # interpolate u(x, t=t^k) from mesh to probe locations
    interpolate = lambda u_t: jnp.interp(x_probes, x_mesh, u_t)
    if u.ndim == 1:
        u_interp = interpolate(u)
    else:
        # apply it to all time instants
        u_interp = jax.vmap(interpolate)(u)
    return u_interp


def _sample_sol_time(u, t_mesh, t_probes):
    interpolate = lambda u: jnp.interp(t_probes, t_mesh, u)
    if u.ndim == 1:
        u_interp = interpolate(u)
    else:
        # apply it to all time instants
        u_interp = jax.vmap(interpolate, in_axes=1, out_axes=1)(u)
    return u_interp


def generate_key(seed):
    if seed is None:
        # JAX doesn't support None as seed value => create a random integer 
        # using NumPy, then pass it to JAX
        # (use the max value storable in a float64)
        seed_new = np.random.default_rng(seed).integers(low=np.iinfo(np.int64).max)
    else:
        seed_new = seed
    key_random = jrn.key(seed_new)
    return key_random


def load_dataset(h5_file_ptr, group_name='/'):
    # NB: the root group name is '/'
    data = {}
    for key, obj in h5_file_ptr[group_name].items():
        if isinstance(obj, h5py.Group):
            data[key] = load_dataset(h5_file_ptr, group_name=obj.name)
        else:
            # obj       h5py.Dataset object, vanishes once file is closed
            # obj[:]    numpy.ndarray, read the entire array, breaks if the
            #               data stored in dataset[key] is a scalar 
            # obj[()]   numpy.ndarray or scalar, read whatever's in the database
            item = obj[()]
            if isinstance(item, h5py.Empty):
                item = None
            elif isinstance(item, bytes):
                # strings stored as b'my string' => turned into actual string
                item = item.decode('utf-8')
            data[key] = item
    return data


def save_dict_to_hdf5(data, group_obj):
    for key in data:
        # call recursively the function till you get to a level where there
        # are no more subdictionaries
        if isinstance(data[key], dict):
            subgroup = group_obj.create_group(key)
            save_dict_to_hdf5(data[key], subgroup)
        elif callable(data[key]):
            # transform the function into an array by pickling it
            group_obj.create_dataset(name=key, 
                                     data=jnp.frombuffer(cloudpickle.dumps(data[key]),
                                                         dtype=jnp.uint8)
                                     )
        elif data[key] is None:
            # "float" empty type
            group_obj.create_dataset(key, data=h5py.Empty('f'))
        else:
            group_obj.create_dataset(key, data=data[key])


def print_optimiser_info(opts):
    text = [f"Optimiser        : {opts['optimiser']}"]
    match opts['optimiser'].lower():
        case 'bfgs':
            pass
        case 'adam' | 'amsgrad':
            text += [f"Learning rate    : {opts['learning_rate']}",
                     f"Max. iterations  : {opts['max_iterations']}"]
        case _:
            raise ValueError(f"Unknown adaptive optimiser '{opts['optimiser']}'")
    print('\n'.join(text))


def generate_mesh(input_data):
    mesh_opts = input_data['mesh']
    sol_file = input_data['files']['solution']
    
    with h5py.File(sol_file, 'r') as file:
        # file['settings']['L']     => HDF5 dataset
        # file['settings']['L'][()] => content of the dataset
        L = file['settings']['L'][()]
        T = file['settings']['T'][()]
        if mesh_opts['dx'] is None:
            print('No x-spacing provided, reading exact solution files for mesh settings...')
            x_mesh = jnp.asarray(file['mesh']['x'])
        else:
            print(f'Calculating space mesh in domain [0, {L}] with discretisation '
                  f"step dx = {mesh_opts['dx']}...")
            # N elements => N+1 nodes
            N_x = int(L / mesh_opts['dx']) + 1
            x_mesh = jnp.linspace(0, L, N_x)
        
        if input_data['steady']:
            warning('Steady simulation, time will be ignored and set to 0 arbitrarily...')
            t_mesh = jnp.array([0.0])
        elif mesh_opts['dt'] is None:
            print('No time step provided, reading exact solution files for mesh settings...')
            t_mesh = jnp.asarray(file['mesh']['t'])
        else:
            print(f'Calculating time step in domain [0, {T}] with time step '
                  f"dt = {mesh_opts['dt']}...")
            # N elements => N+1 nodes
            N_t = int(T / mesh_opts['dt']) + 1
            t_mesh = jnp.linspace(0, T, N_t)
        
        mesh = {'x': x_mesh,
                't': t_mesh
                }
    return mesh


def print_theta(theta_MAP, theta_prior, STEADY, t_mesh=None, t=None):
    # theta is a 2D matrix: a row vector of column vectors
    # [theta(t^1), ..., theta(t^k)]
    # in 1D, it's a single column => make sure that there's a 2nd dimension
    # to avoid breaking the indexing
    theta_prior = jnp.atleast_2d(theta_prior)
    theta_MAP = jnp.atleast_2d(theta_MAP)
    # if 1D, (N,) -> (1,N), but for consistency it should be (N,1)
    if theta_prior.shape[0] == 1:
        theta_prior = theta_prior.T
    if theta_MAP.shape[0] == 1:
        theta_MAP = theta_MAP.T
    
    if STEADY:
        time_str = ''
        idx_time = -1
    else:
        if t < t_mesh[0] or t > t_mesh[-1]:
            raise ValueError('The given time instant does not lie within the time mesh!')
        idx_time = jnp.argmin(jnp.abs(t_mesh - t))
        time_str = f' at t = {t_mesh[idx_time]} s'
    
    theta_report = PrettyTable()
    theta_report.title = f'Theta coefficients{time_str}'
    theta_report.field_names = ['Component (n-th)', 'theta_prior', 'theta_MAP']
    for i in range(theta_MAP.shape[0]):
        theta_report.add_row([i+1, f'{theta_prior[i,idx_time]:+.5e}', f'{theta_MAP[i,idx_time]:+.5e}'])
    check_msg(theta_report.get_string())


def read_input_data(input_data, mode, solver=None, case_type=None):   
    
    match mode.lower():
        case 'field_inversion':
            return _read_field_inversion_input(input_data, input_data['files']['solution'], 
                                               input_data['files']['data'], solver)
        case 'postprocessing':
            return _read_postproc_input(input_data, input_data['FIML_file'], 
                                        input_data['exact_solution_file'], 
                                        input_data['data_file'])
        case 'machine_learning':
            return _read_machine_learning_input(input_data, input_data['files']['training_data'], 
                                                case_type, input_data['files']['test_data'])
        case _:
            raise ValueError("Missing or invalid mode: either 'analysis' or 'postprocessing'")


def _read_field_inversion_input(input_data, exact_sol_file, exp_data_file, solver):
    
    exp_data_file = Path(exp_data_file)
    print(f'Reading experimental data from \n\t{exp_data_file.resolve()}')
    with h5py.File(exp_data_file, 'r') as file:
        # obj       h5py.Dataset object, vanishes once file is closed
        # obj[:]    numpy.ndarray, read the entire array, breaks if the
        #           data stored in dataset[key] is a scalar 
        # obj[()]   numpy.ndarray or scalar, read whatever's in the database
        exp_data = objs.ExperimentalData(x=file['probes']['x'][:],
                                         t=file['probes']['t'][:],
                                         solution=file['exact']['solution'][:])
    
    exact_sol_file = Path(exact_sol_file)
    print(f'Reading exact solution from \n\t{exact_sol_file.resolve()}')
    with h5py.File(exact_sol_file, 'r') as file:
        BC = {'u0': jnp.asarray(file['settings']['u0']), 
              'uL': jnp.asarray(file['settings']['uL']), 
              # HDF5 dataset object -> NumPy array -> lambda function
              'u_IC': cloudpickle.loads(jnp.asarray(file['settings']['u_IC']))
              }
        # HACK: change the way data are stored in CreateRawData to remove this
        # hacky if-else
        if 'baseline' in file:
            nu = cloudpickle.loads(jnp.asarray(file['baseline']['nu']))
        else:
            nu = cloudpickle.loads(jnp.asarray(file['settings']['nu0']))
        
        beta = file['beta']['beta'][:] if 'beta' in file.keys() else None
        exact_sol = objs.ExactSolution(x=file['mesh']['x'][:],
                                       t=file['mesh']['t'][:],
                                       solution=file['exact']['solution'][:],
                                       beta=beta)

    
    problem = objs.Problem(mesh=generate_mesh(input_data), BC=BC, nu_baseline=nu, 
                           solver_fun=solver, STEADY=input_data['steady'])
    
    cost_fun_obj = objs.CostFunction(C_data=input_data['variance_observational'],
                                     C_theta=input_data['variance_prior'],
                                     W_reg=input_data['regularisation_weight'],
                                     data=exp_data)
    
    print('INPUT DATA: ')
    _print_dict(input_data)
    
    return problem, exp_data, exact_sol, cost_fun_obj


def _read_postproc_input(output_data, FIML_file, exact_sol_file, exp_data_file=None):
            
    FIML_file = Path(FIML_file)
    print(f'Reading FIML data from \n\t{FIML_file.resolve()}')
    with open(FIML_file, 'rb') as file:
        FIML = cloudpickle.load(file)
    
    if exp_data_file is not None:
        exp_data_file = Path(exp_data_file)
        print(f'Reading experimental data from \n\t{exp_data_file.resolve()}')
        with h5py.File(exp_data_file, 'r') as file:
            # obj       h5py.Dataset object, vanishes once file is closed
            # obj[:]    numpy.ndarray, read the entire array, breaks if the
            #           data stored in dataset[key] is a scalar 
            # obj[()]   numpy.ndarray or scalar, read whatever's in the database
            exp_data = objs.ExperimentalData(x=file['probes']['x'][:],
                                             t=file['probes']['t'][:],
                                             solution=file['exact']['solution'][:])
    else:
        print('No experimental data file given, using the experimental data '
              'stored in the FIML object (FIML.exp_data)...')
        exp_data = FIML.exp_data
    
    exact_sol_file = Path(exact_sol_file)
    print(f'Reading exact solution from \n\t{exact_sol_file.resolve()}')
    with h5py.File(exact_sol_file, 'r') as file:
# =============================================================================
#         BC = {'u0': jnp.asarray(file['settings']['u0']), 
#               'uL': jnp.asarray(file['settings']['uL']), 
#               # HDF5 dataset object -> NumPy array -> lambda function
#               'u_IC': cloudpickle.loads(jnp.asarray(file['settings']['u_IC']))
#               }
#         nu = cloudpickle.loads(jnp.asarray(file['baseline']['nu']))
# =============================================================================
        
        beta = file['beta']['beta'][:] if 'beta' in file.keys() else None
        exact_sol = objs.ExactSolution(x=file['mesh']['x'][:],
                                       t=file['mesh']['t'][:],
                                       solution=file['exact']['solution'][:],
                                       beta=beta)
    
    print('POSTPROCESSING DATA: ')
    _print_dict(output_data)
    
    return FIML.problem, exact_sol, exp_data, FIML


def _read_machine_learning_input(input_data, train_file, case_type, test_file=None, test2tot=0.7):
    print(f'Reading training data from \n\t{train_file.resolve()}')
    with h5py.File(train_file, 'r') as file:
        data = []
        for case_name in file:
            case_obj = cloudpickle.loads(jnp.asarray(file[case_name]))
            params = _read_dir_name(case_name, case_type)
            sim_obj = objs.SimulationML(case_id=case_name, params=params,
                                        FI=case_obj)
            data.append(sim_obj)
    if test_file is None:
        warning('No test data file explicitly passed, splitting randomly training data '
                f'into {100*test2tot:.0f}% training data and {100*(1-test2tot):.0f}% test data...')
        N_test = int((1-test2tot) * len(data))  # truncated (same as floor)
        if N_test == 0:
            warning(f'{100*(1-test2tot):.0f}% of data is rounded to 0, forcing it to 1 test case...\n'
                    '(Either increment the number of data, or modify the fraction for testing)')
        # randomly shuffle the data and remove the last 30% of them => same as
        # randomly sampling 30%, if the order is not relevant
        # https://stackoverflow.com/a/44884212/17220538
        random.shuffle(data)
        test_data = data[-N_test:]              # last "N_test" elements
        train_data = data[0:len(data)-N_test]   # all the others
    else:
        train_data = data
        
        print(f'Reading test data from \n\t{test_file.resolve()}')
        with h5py.File(test_file, 'r') as file:
            test_data = []
            for case_name in file:
                MAP_obj = cloudpickle.loads(jnp.asarray(file[case_name]))
                params = _read_dir_name(case_name, case_type)
                sim_obj = objs.SimulationML(case_id=case_name, params=params,
                                            FI=MAP_obj)
                test_data.append(sim_obj)
    
    ML_obj = objs.DataML(training=tuple(train_data), test=tuple(test_data))
    return ML_obj


def _read_dir_name(case_name, case_type):
    match case_type:
        case 'Burgers_u0uLnu0beta':
            # (?P<u0>...)   named capture groupd with name 'u0'
            pattern = re.compile(r'^u0(?P<u0>[-+]?\d*\.?\d+)'
                                 r'_uL(?P<uL>[-+]?\d*\.?\d+)'
                                 r'_nu0(?P<nu0>[-+]?\d*\.?\d+)'
                                 r'_(?P<name>.+)$'
                                 )
            match = pattern.fullmatch(case_name)
            if match is None:
                warning(f"Case directory '{case_name}' doesn't match the given pattern")
            params = objs.BurgersParameters(u0=float(match.group('u0')),
                                            uL=float(match.group('uL')),
                                            nu0=float(match.group('nu0')),
                                            beta_type=match.group('name'))
        case _:
            raise ValueError(f"Unknown case dir {case_type}")
    return params



def _print_dict(input_dict, prefix=''):
    for key in input_dict:
        if isinstance(input_dict[key], dict):
            print(f'{key}:')
            _print_dict(input_dict[key], prefix='\t')
        else:
            # print reserving 20 characters (whitespace-filled, it not all used)
            print(f'{prefix}{key:<20}: {input_dict[key]}')


def _read_file_or_obj(input_obj, msg=''):
    if isinstance(input_obj, Path) or isinstance(input_obj, str):
        input_obj = Path(input_obj)
        print(f'Reading {msg} \n\t{input_obj.resolve()}')
        with open(input_obj, 'r') as file:
            match input_obj.suffix:
                case '.yaml':
                    input_data = yaml.safe_load(file)
                case '.pkl':
                    input_data = cloudpickle.load(file)
                case _:
                    raise ValueError(f"Unknown file extension {input_obj.suffix}")
    elif isinstance(input_obj, dict):
        input_data = input_obj.copy()
    else:
        raise ValueError('Unknown input object type: neither Path, nor str')
    return input_data


def pack_post_data(FIML=None, sol_file=None, exp_file=None):
    
    if FIML is not None:
        return FIML
    
    if sol_file is not None and exp_file is not None:
        sol_file = Path(sol_file)
        print(f'Reading FIML solution from \n\t{sol_file.resolve()}')
        # https://stackoverflow.com/a/41586571/17220538
        with h5py.File(sol_file, 'r') as file:
            print('Available groups (left) and their datasets (right): ')
            for group_name, group in file.items():
                print('\t', group_name, '\t', list(group.keys()))
            
            prior_data = load_dataset(file, 'prior')
            posterior_data = load_dataset(file, 'posterior')
            MAP_data = load_dataset(file, 'MAP')
            mesh = load_dataset(file, 'mesh')
            input_data = load_dataset(file, 'input_data')
            BC = load_dataset(file, 'boundary_conditions')
            
            prior = SimpleNamespace(
                x=jnp.asarray(mesh['x']),
                t=jnp.asarray(mesh['t']),
                solution=jnp.asarray(prior_data['solution']),
                beta=jnp.asarray(prior_data['beta']))
            posterior = SimpleNamespace(
                x=jnp.asarray(mesh['x']),
                t=jnp.asarray(mesh['t']),
                solution=jnp.asarray(posterior_data['solution']),
                beta=jnp.asarray(posterior_data['beta']))
            MAP = SimpleNamespace(
                x=jnp.asarray(mesh['x']),
                t=jnp.asarray(mesh['t']),
                solution=jnp.asarray(MAP_data['solution']),
                beta=jnp.asarray(MAP_data['beta']))
        
        exp_file = Path(exp_file)
        print(f'Reading experimental data from \n\t{exp_file.resolve()}')
        with h5py.File(exp_file, 'r') as file:
            exp_data = load_dataset(file)
            exp = objs.ExperimentalData(x=exp_data['x'], t=exp_data['t'], 
                                        solution=exp_data['solution'])
    
    return prior, posterior, exp, MAP, BC, input_data
