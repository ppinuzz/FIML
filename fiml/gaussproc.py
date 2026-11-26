"""
Utilities for Gaussian Processes.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import optax
from time import perf_counter
import scipy.optimize as sciopt
import fiml.auxiliaries as aux
import fiml.objects as objs
import numpy as np


def fit_gauss_process(train_data, beta_mean=1, kernel_name='RBF_Gauss',
                      learn_rate=1e-2, max_iter=500, tol=1e-5, method='adam',
                      every_n_iter=40):
    
    # normalise input U to remove scale effects
    u_mean = jnp.mean(train_data.sol)
    u_std = jnp.std(train_data.sol)
    U = (train_data.sol - u_mean) / u_std
    
    # log likelihood:
    # log[P(beta | {amp_kern, len_kern, sigma_err})] = 
    #   - 0.5 N ln(2*pi) - 0.5 ln(det(Sigma)) - 0.5 (beta-beta_mean) * Sigma^-1 * (beta-beta_mean)^T
    # => log to make sure they will always stay positive in the optimiser
    # (if the variable X is representing log(x), x will always be > 0 since it
    # is x = e^X)
    amp_kern_log = jnp.maximum(jnp.sqrt(jnp.mean((train_data.beta - beta_mean)**2)), 1e-3)
    len_kern_log = 1
    sigma_err = 0.05 * amp_kern_log
    P0 = jnp.log(jnp.array([amp_kern_log, len_kern_log, sigma_err]))
    
    # C_beta's are independent matrices, one per each subdataset => assemble
    # them into a block diagonal total covariance matrix
    C_beta = jsl.block_diag(*train_data.covariance_beta_blocks)
    
    @jax.jit
    def negative_log_likelihood(P):
        amp_kernel = jnp.exp(P[0])
        len_scale_kernel = jnp.exp(P[1])
        sigma_err = jnp.exp(P[2])
        Sigma = covariance_GP(kernel, (U, U, kernel_name, (amp_kernel, len_scale_kernel)), C_beta, sigma_err)
        # log likelihood is negative => maximise negative = minimise positive
        NLL = - log_likelihood(train_data.beta, beta_mean, Sigma)
        return NLL
    
    NLL_and_grad = jax.jit(jax.value_and_grad(negative_log_likelihood))
    
    match method.lower():
        case 'adam':
            optimiser = optax.adam(learning_rate=learn_rate)
        case 'amsgrad':
            optimiser = optax.amsgrad(learning_rate=learn_rate)
        case _:
            raise ValueError(f"Unknown adaptive optimiser '{method}'")
    
    # initialise optimiser internal state with estimates for the 1st and 
    # 2nd moments
    opt_state = optimiser.init(P0)
    history = {'P': jnp.full((max_iter+1, P0.shape[0]), jnp.nan),
               'minus_log_likelihood': jnp.full((max_iter+1, ), jnp.nan),
               'grad': jnp.full((max_iter+1, P0.shape[0]), jnp.nan),
               'grad_norm': jnp.full((max_iter+1, ), jnp.nan)
               }
    
    t_start = perf_counter()
    n = 0
    P_n = P0
    grad_norm_n = tol + 1
    while n < max_iter and grad_norm_n > tol :
        NLL_n, grad_n = NLL_and_grad(P_n)
        if ~jnp.isfinite(NLL_n):
            raise ValueError(f'Non-finite log likelihood at iteration {n}, guess: {list(P_n)}')
        if ~(jnp.isfinite(grad_n).all()):
            raise ValueError(f'Non-finite log likelihood gradient at iteration {n}, guess: {list(P_n)}')
        
        grad_norm_n = jnp.linalg.norm(grad_n)
        history['P'] = history['P'].at[n].set(jnp.exp(P_n))
        history['minus_log_likelihood'] = history['minus_log_likelihood'].at[n].set(NLL_n)
        history['grad'] = history['grad'].at[n].set(grad_n)
        history['grad_norm'] = history['grad_norm'].at[n].set(grad_norm_n)
        
        if n % every_n_iter == 0:
            print(f'ADAM iteration {n:5d}: NLL = {NLL_n:.8e}, ||grad(NLL)|| = {grad_norm_n:.5e}')
        
        updates, opt_state = optimiser.update(grad_n, opt_state, P_n)
        P_n = optax.apply_updates(P_n, updates)
        n += 1
    
    if grad_norm_n > tol:
        converged = False
        status = 1
        message = (f'||grad(log_likelihood)|| = {grad_norm_n:.5e} > tolerance '
                   f'after exhausting the {max_iter} iterations')
        aux.warning(message)
    else:
        converged = True
        status = 0
        message = f'Converged in {n} iterations'
    
    t_elapsed = perf_counter() - t_start
    print(f'Time required to evaluate cost function jacobian: {t_elapsed:.3e} s')

    # remove unused history entries (left to NaN)
    for entry in history:
        history[entry] = history[entry][0:n+1]
    
    # variables are the log() => return the actual value, not the log
    P_n_opt = jnp.exp(P_n)
    NLL_n, grad_n = NLL_and_grad(P_n)
    if ~jnp.isfinite(NLL_n):
        raise ValueError(f'Non-finite log likelihood at iteration {n}, guess: {list(P_n)}')
    if ~(jnp.isfinite(grad_n).all()):
        raise ValueError(f'Non-finite log likelihood gradient at iteration {n}, guess: {list(P_n)}')
    history['P'] = history['P'].at[n].set(jnp.exp(P_n))
    history['minus_log_likelihood'] = history['minus_log_likelihood'].at[n].set(NLL_n)
    history['grad'] = history['grad'].at[n].set(grad_n)
    history['grad_norm'] = history['grad_norm'].at[n].set(jnp.linalg.norm(grad_n))
    
    
    Sigma = covariance_GP(kernel, (U, U, kernel_name, (P_n_opt[0], P_n_opt[1])), C_beta, P_n_opt[2])
    GP = objs.GaussianProcess(data=train_data, 
                              kernel=lambda x, z, params: kernel(x, z, kernel_name, params),
                              log_likelihood=log_likelihood,
                              beta_mean=beta_mean, training_covariance=Sigma,
                              parameters=tuple(P_n_opt),
                              normalisation_mean=float(u_mean),
                              normalisation_std=float(u_std)
                              )
    
    # for the function checking the result
    # NB: jac contains the gradient of the LOG function w.r.t. the LOGS of the
    # parameters
    result = sciopt.OptimizeResult(x=np.asarray(P_n), fun=float(NLL_n),
                                   jac=np.asarray(grad_n), nit=int(n),
                                   # +1 to count the intial guess
                                   nfev=int(n+1), njev=int(n+1),
                                   success=bool(converged), status=status, 
                                   message=message, history=history)
    
    return GP, result


def kernel(x, z, name, params):
    match name:
        case 'RBF_Gauss':
            amp, h = params
            k = amp**2 * _kernel_RBF_Gauss(x, z, h)
        case _:
            raise ValueError(f"Unknown kernel '{name}'")
    return k


def _kernel_RBF_Gauss(x, x_c, h):
    # x has shape (N, ) => x[:,jnp.newaxis] - x_c[jnp.newaxis,:] has shape (N,N)
    # => all pairwise differences (x_i - x_j), con i,j covering the whole range
    # i,j = 0, ..., N-1
    k = jnp.exp(- 0.5 * (x[:,jnp.newaxis] - x_c[jnp.newaxis,:])**2 / h**2)
    return k


def covariance_GP(kernel_fun, kernel_params, C_beta, sigma_s, epsilon=1e-6):
    K = kernel_fun(*kernel_params)
    Sigma = K + C_beta + (sigma_s**2 + epsilon) * jnp.eye(C_beta.shape[0])
    return Sigma


def log_likelihood(beta, beta_mean, Sigma):
    N = Sigma.shape[0]
    L1 = 0.5 * N * jnp.log(2*jnp.pi)
    R_chol = jnp.linalg.cholesky(Sigma, upper=False)
    # since Sigma = R*R^T, and R is lower diagonal, det(Sigma) = det(R)**2
    # - 0.5*ln(det(Sigma))
    L2 = jnp.sum(jnp.log(jnp.diag(R_chol)))
    # x^T * A^-1 * x = x^T * y, where y is the solution of Ay = x
    # since I'm already calculating Cholesky's decomposition of Sigma
    # (True: the matrix passed is the lower tridiagonal, not the upper tridiagonal)
    y = jsl.cho_solve((R_chol, True), beta-beta_mean)
    L3 = 0.5 * (beta-beta_mean).T @ y
    L = - (L1 + L2 + L3)
    return L