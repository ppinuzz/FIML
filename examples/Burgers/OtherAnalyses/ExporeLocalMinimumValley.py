#!/usr/bin/env python3
"""
Study the existence of possible multiple valleys in the cost function J(theta).

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
from prettytable import PrettyTable

pkg = Path(r'C:\Andrea\Dottorato\FIML')
burgers_pkg = pkg / 'examples' / 'Burgers'

for path in (pkg, burgers_pkg):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import fiml
# required when loading the FIML object
import burgers


#%% INPUT DATA AND FILES

input_file = burgers_pkg / 'setup_FIML/BCDirichlet_steady_nu_u.yaml'
FIML_results = burgers_pkg / 'solution_FIML/BCDirichlet_steady_nu_u/RBF_Gauss_x_poly_softplus/FIML_object.pkl'
out_dir = FIML_results.parent / 'valleyStudy'

# perturbation applied as a radius in the N_theta-dimensional space of theta 
# (e.g. R^11 if theta has 11 components => the perturbation is a radius in R^11)
delta = jnp.array([0, 10, 20, 40, 80, 160])

RANDOM_KEY = 123


#%% READ DATA

with open(FIML_results, 'rb') as file:
    FIML = cloudpickle.load(file)

with open(input_file, 'r') as file:
    input_data = yaml.safe_load(file)
    optimiser_opts = input_data['optimisation']
    if optimiser_opts['optimiser'].lower() != 'adam':
        raise ValueError('This code only works with ADAM!')

out_dir.mkdir(parents=True, exist_ok=True)


#%% SOLVE MINIMISATION PROBLEM WITH ADDED NOISE


#%% SET UP COST FUNCTION

# original cost function
cost_function = FIML.cost

def cost_with_aux(theta):
    cost, success = cost_function(theta)
    return cost, (cost, success)

@jax.jit
def cost_and_jacobian(theta):
    # has_aux=True specifies that the function returns (f(theta), auxiliaries)
    # and that it should be differentiated only w.r.t. the 1st output
    # (the actual numerical output)
    grad, (cost, success) = jax.jacfwd(cost_with_aux, has_aux=True)(theta)
    return cost, grad, success


#%% REFERENCE CONDITION (MAP FROM FIML CODE)

theta_prior = jnp.zeros_like(FIML.MAP.theta)
cost_MAP, jacob_MAP, success = cost_and_jacobian(FIML.MAP.theta)
if not success:
    raise ValueError('Evaluating cost and jacobian at MAP point did not succeed!')

print(f'Prior: theta_0 = {theta_prior}')
print('Reference solution: MAP point')
print(f'theta_MAP = {FIML.MAP.theta}')
print(f'cost(theta_MAP) = {cost_MAP}')
print(f'grad(J) = {jacob_MAP}')
print(f'||grad(J)|| = {jnp.linalg.norm(jacob_MAP)}')


#%% CREATE RANDOM DIRECTIONS

# theta_new = theta_MAP + delta * ||theta_MAP|| * vet_random
# where vet_random is a random unit vector in R^N_theta (e.g. R^11)
key = jax.random.PRNGKey(RANDOM_KEY)
# each column is a different (not-yet-unit) vector
vectors = jax.random.uniform(key, shape=(theta_prior.size, delta.size), minval=-1, maxval=1)
# keepdims=True     return a norm vector (1, N) rather than (N, )
# (required to have the correct element-wise division)
vectors = vectors / jnp.linalg.norm(vectors, axis=0, keepdims=True)

report = PrettyTable()
report.title = 'Restarting ADAM with x0 = x_opt * (1 + delta)'
x0 = []
theta_found = []
report.field_names = ['delta', 'error [%]', 'J(theta)', '||grad(J)||', 'success']
for i, delta_i in enumerate(delta):
    success = True
    print(f'DELTA: {delta_i}')

    theta_i = FIML.MAP.theta + delta_i*jnp.linalg.norm(FIML.MAP.theta)*vectors[:,i]

    opt = fiml.fiml.minimise_cost_function(cost_and_jacobian,
                                           x0=theta_i,
                                           method=optimiser_opts['optimiser'],
                                           learn_rate=optimiser_opts['learning_rate'],
                                           max_iter=optimiser_opts['max_iterations']
                                           )
    try:
        fiml.auxiliaries.check_optimiser_results(opt)
    except:
        success = False
    fiml.auxiliaries.print_theta(opt.x, jnp.zeros_like(theta_i), FIML.problem.STEADY, FIML.problem.t_mesh)
    cost_i, grad_i, success_i = cost_and_jacobian(opt.x)
    success = success and success_i
    err = jnp.linalg.norm(jnp.abs(opt.x - FIML.MAP.theta) / FIML.MAP.theta)
    
    print(f'theta = {theta_i}')
    print(f'cost(theta) = {cost_i}')
    print(f'grad(J) = {grad_i}')
    print(f'||grad(J)|| = {jnp.linalg.norm(grad_i)}')
    
    
    x0.append(theta_i)
    theta_found.append(opt.x)
    report.add_row([f'{delta_i}', f'{100*err:.2f}', f'{cost_i:.5e}', f'{jnp.linalg.norm(grad_i):.5e}', f'{success}'])

print(report)