"""
Burgers' solver input file:
    *) u(x,0) = sin(pi x/L)
    *) homogeneous Dirichlet BCs
    *) nu(x, u, du/dx)

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


#%% DOMAIN 

# leave set to 1 to represent the non-dimensional domain
L = 1
T = 1

x0 = 0
xL = L


#%% BOUNDARY + INITIAL CONDITIONS

u0 = 0
uL = 0
u_IC = lambda x: jnp.sin(jnp.pi*x/L)


#%% DISCRETISATION

# uniform mesh
dt = 3.125e-4
dx = 7.813e-5


#%% VISCOSITY

nu0 = 1e-2

# base model: constant viscosity
nu0_fun = lambda x, t, u, ux, uxx: nu0

# actual viscosity 
# FIXME: current signature contains also u_xx, but it's NOT implemented a 
# dependence on u_xx, as the Jacobian wouldn't be tridiagonal => use GMRES and
# a different solver (currently not implemented)
#nu = lambda x, t, u, ux, uxx: nu0 * (1 + 20*jnp.sin(jnp.pi*x/L)*(1-jnp.exp(-t)))
nu = lambda x, t, u, ux, uxx: nu0 * (1 + 20*jnp.sin(jnp.pi*x/L)*(1-jnp.exp(-jnp.abs(u))) + 0.5*jnp.abs(ux))


#%% EXACT BETA CORRECTION

# nu_exact = beta * nu_approx
# here, nu_approx = nu0_fun = nu0, therefore beta = nu / nu0
beta = lambda x, t, u, ux, uxx: nu(x, t, u, ux, uxx) / nu0_fun(x, t, u, ux, uxx)


#%% PROBES

N_probes = 10
# 1st and last probes are inside the domain, not on the boundaries x=0 and x=L
dx_in_out = 0.05
x_probes = jnp.linspace(dx_in_out, L-dx_in_out, N_probes)