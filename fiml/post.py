"""
Postprocessing utilities for FIML.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import jax
# use float64, rather than float32
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import fiml.auxiliaries as aux
from prettytable import PrettyTable


def check_experiments_in_sigma_band(sol, exp, x, t, x_exp, t_exp, ax_sol=2, 
                                    N_sigma=2, t_plot=None, fig_file=None, 
                                    PLOT=True, VIDEO=False):
    sol = jnp.asarray(sol)
    # for each space-time point, calculate the sample mean of the solutions
    # obtained solving the equation for each sample
    # (1st index: space, 2nd index: time, 3rd index: sampling)
    u_mean = jnp.mean(sol, axis=ax_sol)
    u_sigma = jnp.std(sol, axis=ax_sol)
    u_upper = u_mean + N_sigma*u_sigma
    u_lower = u_mean - N_sigma*u_sigma
    
    # if no time instant to plot is given, the last one is assumed
    # (where discrepancy is likely to be the biggest)
    if t_plot is None:
        idx_plot_exp = -1
        aux.warning('No time instant given, the last instant '
                    f't = {t_exp[idx_plot_exp]} s will be plotted...')
    else:
        if t_plot < t_exp[0] or t_plot > t_exp[-1]:
            raise ValueError('Given time instant lies outside of the experimental range: \n'
                             f'Given time: {t_plot} s\n'
                             f'Experimental range: {t_exp[0]}-{t_exp[-1]} s')
        # find experimental point closest to the given time instant
        idx_plot_exp = jnp.argmin(jnp.abs(t_exp - t_plot))
    # interpolate numerical solution to the chosen experimental point 
    # (do NOT interpolate experimental data, as that could be meaningless!)
    t_exp_plot = t_exp[idx_plot_exp]
    u_mean_t = aux.sample_solution(u_mean, t_mesh=t, t_probes=t_exp)
    u_up_t = aux.sample_solution(u_upper, t_mesh=t, t_probes=t_exp)
    u_low_t = aux.sample_solution(u_lower, t_mesh=t, t_probes=t_exp)
    
    # interpolate numerical data at probe locations to check whethet experimental
    # data lie withing +/- 2*sigma from the sample mean
    u_upper_probes = aux.sample_solution(u_upper, x_mesh=x, t_mesh=t, x_probes=x_exp, t_probes=t_exp)
    u_lower_probes = aux.sample_solution(u_lower, x_mesh=x, t_mesh=t, x_probes=x_exp, t_probes=t_exp)
    
    lies_within = (exp <= u_upper_probes) & (exp >= u_lower_probes)
    N_within_t = jnp.count_nonzero(lies_within[idx_plot_exp,:])
    N_probes = lies_within[idx_plot_exp,:].size
    fraction_within_t = N_within_t / N_probes
    aux.check_msg(f'{100*fraction_within_t:.2f}% ({N_within_t}/{N_probes}) of the '
                  f'experimental data at time t = {t_exp_plot:.5f} s '
                  f'lie within +/- {N_sigma} standard deviations from the sample mean')
    fraction_within_total = jnp.count_nonzero(lies_within) / lies_within.size
    aux.check_msg(f'{100*fraction_within_total:.2f}% of the experimental data point (x,t) '
                  f'lie within +/- {N_sigma} standard deviations from the sample mean')
    
    # interpolate solution at t = t_plot on probe locations
    u_mean_probes = aux.sample_solution(u_mean_t[idx_plot_exp,:], x_mesh=x, x_probes=x_exp)
    delta_u = u_mean_probes - exp[idx_plot_exp,:]
    err = delta_u / exp[idx_plot_exp,:]
    
    
    err_report = PrettyTable()
    err_report.title = f'Mean of the distribution vs Experimental data at t = {t_exp_plot:.5f} s'
    err_report.field_names = ['x', 'Mean', 'Experiment', 'Difference', 'Relative error [%]']
    for k in range(exp.shape[1]):
        err_report.add_row([f'{x_exp[k]:.5f}', f'{u_mean_probes[k]:.5f}',
                            f'{exp[idx_plot_exp,k]:.5f}', f'{delta_u[k]:.5f}', 
                            f'{100*err[k]:.2f}'])
    aux.check_msg(err_report.get_string())
    
    if PLOT:
        # data are stored as [t^k, u_1^k, ..., u_N^k] => create a column vector of
        # column vectors => flatten the 2D array in column-major order (Fortran-like)
        #sol_exp_t = jnp.ravel(sol_exp_interp_t.T)
        
        fig = plt.figure()
        plt.plot(x, u_mean_t[idx_plot_exp,:], label='Sample mean', color='r')
        plt.fill_between(x, u_up_t[idx_plot_exp,:], u_low_t[idx_plot_exp,:], alpha=0.3, color='r')
        #plt.plot(x, u_up_t[idx_plot_exp,:], color='r')
        #plt.plot(x, u_low_t[idx_plot_exp,:], color='r')
        plt.plot(x_exp, exp[idx_plot_exp,:], '*', label='Experiment')
        plt.xlabel('$x$ [-]')
        plt.ylabel('$u$ [-]')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend()
        plt.title(fr'Sample mean solution with $\pm 2\sigma$ at $t = {t_exp_plot:.5f}$ s')
        plt.show(block=False)
        if fig_file is not None:
            aux.save_fig(fig, fig_file.name, fig_file.parent)


def compare_exact_beta(beta_num, beta_exact, x, t, x_beta, t_beta, 
                       t_plot=None, fig_file=None, PLOT=True, VIDEO=False,
                       beta_labels=('', )):
    
    beta_exact = jnp.asarray(beta_exact)
    x = jnp.asarray(x)
    t = jnp.asarray(t)
    x_beta = jnp.asarray(x_beta)
    t_beta = jnp.asarray(t_beta)
    STEADY = False
    if t.size == 1:
        aux.warning('Time mesh contains only one time instant, treating the '
                    'problem as steady...')
        STEADY = True
    # if no time instant to plot is given, the last one is assumed
    # (where discrepancy is likely to be the biggest)
    if not STEADY:
        if t_plot is None:
            idx_plot_beta = -1
            aux.warning('No time instant given, the last instant '
                        f't = {t_beta[idx_plot_beta]} s will be plotted...')
        else:
            if t_plot < t_beta[0] or t_plot > t_beta[-1]:
                raise ValueError('Given time instant lies outside of the time range '
                                 'of the reference exact beta: \n'
                                 f'Given time: {t_plot} s\n'
                                 f'Experimental range: {t_beta[0]}-{t_beta[-1]} s')
            # find exact_beta point closest to the given time instant
            idx_plot_beta = jnp.argmin(jnp.abs(t_beta - t_plot))
    else:
        # there's only one time instant: the last one
        idx_plot_beta = -1
    t_beta_plot = t_beta[idx_plot_beta]
    
    # make sure beta is ordered like the solution (Nt, Nx)
    if beta_exact.shape == (x_beta.size, t_beta.size):
        beta_exact = beta_exact.T
    
    # beta_num can either be
    #   - a single array, to check a single beta vs the exact one
    #   - a tuple of arrays (e.g. prior, posterior and MAP)
    if isinstance(beta_num, tuple):
        if len(beta_num) == 1:
            beta_num = jnp.asarray(beta_num)
        else:
            beta_num = tuple([jnp.asarray(beta_i) for beta_i in beta_num])
    else:
        beta_num = (jnp.asarray(beta_num), )

    if not isinstance(beta_labels, tuple):
        beta_labels = tuple(beta_labels)
    if len(beta_labels) != len(beta_num):
        raise ValueError('If labels are provided, each beta object must have a label!')
    
    # create figure out of the loop, then update it
    if PLOT:
        fig_beta, ax_beta = plt.subplots()
        ax_beta.plot(x_beta, beta_exact[idx_plot_beta,:], 'k--', label=r'Exact $\beta$')
        ax_beta.set_xlabel(r'$x$ [-]')
        ax_beta.set_ylabel(r'$\beta$ [-]')
        ax_beta.autoscale(enable=True, axis='x', tight=True)
        time_str = f'at $t = {t_beta_plot:.5f}$ s' if not STEADY else ''
        ax_beta.set_title(fr'$\beta$ correction {time_str}')
        
        fig_err, ax_err = plt.subplots()
        ax_err.set_xlabel(r'$x$ [-]')
        ax_err.set_ylabel(r'$(\beta - \beta_\mathrm{ex})/\beta_\mathrm{ex}$ [\%]')
        ax_err.autoscale(enable=True, axis='x', tight=True)
        ax_err.set_title(fr'Error on $\beta$ {time_str}')
    
    for i, beta_i in enumerate(beta_num):
        # beta originally is transposed because of how it's formulated as a matrix
        # product (i.e. Nx x Nt, not Nt x Nx like the solution matrix)
        if beta_i.ndim == 3:
            # (average as beta_num is a 2D space-time field for each sample => 3D array)
            beta_i_mean = jnp.mean(beta_i, axis=2)
        else:
            beta_i_mean = beta_i
        if beta_i_mean.shape == (x.size, t.size):
            beta_i_mean = beta_i_mean.T
    
        # numerical beta (from solver) reinterpolated over the space-time grid
        # of the exact beta (to avoid distorting the exact beta)
        beta_xt = aux.sample_solution(beta_i_mean, x_mesh=x, x_probes=x_beta, 
                                      t_mesh=t, t_probes=t_beta)
        
        err_beta = (beta_xt - beta_exact) / beta_exact
        # average of the relative errors, in modulus
        MAE_err_t = jnp.mean(jnp.abs(err_beta[idx_plot_beta]))
        aux.check_msg(f'Beta object: {beta_labels[i]}\n'
                      f'At t = {t_beta_plot:.5f}:\n'
                      f'\tAverage of the relative errors, in modulus: {100*MAE_err_t:.2f}%\n'
                      f'\tMaximum error: {100*jnp.max(err_beta[idx_plot_beta]):.2f}%\n'
                      f'\tMinimum error: {100*jnp.min(err_beta[idx_plot_beta]):.2f}%')
    
        if PLOT:
            line, = ax_beta.plot(x_beta, beta_xt[idx_plot_beta,:], 
                                 label=rf'{beta_labels[i]} $\beta$')
            line, = ax_err.plot(x_beta, 100*err_beta[idx_plot_beta,:],
                                label=rf'{beta_labels[i]}')
            
        if VIDEO:
            fig, ax = plt.subplots()
            line_num, = ax.plot(x_beta, jnp.zeros_like(x_beta), label=r'Calculated $\beta$')
            line_exact, = ax.plot(x_beta, jnp.zeros_like(x_beta), label=r'Exact $\beta$')
            ax.legend(loc='upper right')
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(0, 2)
            # enable interactive mode
            plt.ion()
            plt.show(block=False)
            for j in range(t_beta.shape[0]):
                # stop animation if the user closed the window
                if not plt.fignum_exists(fig.number):
                    break
                line_num.set_ydata(beta_xt[j,:])
                line_exact.set_ydata(beta_exact[j,:])
                fig.canvas.draw_idle()
                #fig.canvas.flush_events()
                #plt.pause(0.0000001)
                plt.pause(0.01)
                ax.set_title(f'$t = {t[j]:.4f}$ s')
            plt.ioff()
            # show figure only if it hasn't been closed by the user
            if plt.fignum_exists(fig.number):
                plt.show()

    if PLOT:
        ax_beta.legend()
        fig_beta.show()
        if fig_file is not None:
            aux.save_fig(fig_beta, f'{fig_file.name}_{beta_labels[i]}', fig_file.parent)
        
        ax_err.legend()
        fig_err.show()
        if fig_file is not None:
            aux.save_fig(fig_err, f'{fig_file.name}_{beta_labels[i]}_error', fig_file.parent)
        
        

def compare_prior_posterior(prior, posterior, exp, x, t, x_exp, t_exp, ax_sol=2,
                            N_sigma=2, t_plot=None, fig_file=None, 
                            PLOT=True, VIDEO=False):
    
    # for each space-time point, calculate the sample mean of the solutions
    # obtained solving the equation for each sample
    # (1st index: space, 2nd index: time, 3rd index: sampling)
    u_mean_prior = jnp.mean(prior, axis=ax_sol)
    u_sigma_prior = jnp.std(prior, axis=ax_sol)
    u_upper_prior = u_mean_prior + N_sigma*u_sigma_prior
    u_lower_prior = u_mean_prior - N_sigma*u_sigma_prior
    
    u_mean_post = jnp.mean(posterior, axis=ax_sol)
    u_sigma_post = jnp.std(posterior, axis=ax_sol)
    u_upper_post = u_mean_post + N_sigma*u_sigma_post
    u_lower_post = u_mean_post - N_sigma*u_sigma_post
    
    # if no time instant to plot is given, the last one is assumed
    # (where discrepancy is likely to be the biggest)
    if t_plot is None:
        idx_plot_exp = -1
        aux.warning('No time instant given, the last instant '
                    f't = {t_exp[idx_plot_exp]} s will be plotted...')
    else:
        if t_plot < t_exp[0] or t_plot > t_exp[-1]:
            raise ValueError('Given time instant lies outside of the experimental range: \n'
                             f'Given time: {t_plot} s\n'
                             f'Experimental range: {t_exp[0]}-{t_exp[-1]} s')
        # find experimental point closest to the given time instant
        idx_plot_exp = jnp.argmin(jnp.abs(t_exp - t_plot))
    # actual time instant plotted 
    t_exp_plot = t_exp[idx_plot_exp]
    # interpolate numerical solution at that time instant
    u_mean_prior_exp = aux.sample_solution(u_mean_prior, x_mesh=x, t_mesh=t, t_probes=t_exp_plot)
    u_upper_prior_exp = aux.sample_solution(u_upper_prior, x_mesh=x, t_mesh=t, t_probes=t_exp_plot)
    u_lower_prior_exp = aux.sample_solution(u_lower_prior, x_mesh=x, t_mesh=t, t_probes=t_exp_plot)
    u_mean_post_exp = aux.sample_solution(u_mean_post, x_mesh=x, t_mesh=t, t_probes=t_exp_plot)
    u_upper_post_exp = aux.sample_solution(u_upper_post, x_mesh=x, t_mesh=t, t_probes=t_exp_plot)
    u_lower_post_exp = aux.sample_solution(u_lower_post, x_mesh=x, t_mesh=t, t_probes=t_exp_plot)
    
    if PLOT:
        fig = plt.figure()
        plt.plot(x, u_mean_prior_exp, label='Prior', color='r')
        plt.fill_between(x, u_upper_prior_exp, u_lower_prior_exp, alpha=0.3, color='r')
        #plt.plot(x, u_upper_prior[idx_plot_exp,:], color='r')
        #plt.plot(x, u_lower_prior[idx_plot_exp,:], color='r')
        
        plt.plot(x, u_mean_post_exp, label='Posterior', color='b')
        plt.fill_between(x, u_upper_post_exp, u_lower_post_exp, alpha=0.3, color='b')
        #plt.plot(x, u_upper_post[idx_plot_exp,:], color='b')
        #plt.plot(x, u_lower_post[idx_plot_exp,:], color='b')
        
        plt.plot(x_exp, exp[idx_plot_exp,:], 'k*', label='Experiment')
        plt.xlabel('$x$ [-]')
        plt.ylabel('$u$ [-]')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend()
        plt.title(fr'Prior vs Posterior at $t = {t_exp_plot:.5f}$ s')
        plt.show(block=False)
        if fig_file is not None:
            aux.save_fig(fig, fig_file.name, fig_file.parent)


def check_parametrisation_fitness(beta, B_x, B_t, BETA_MIN=1e-2):
    # beta = F(Bx^T * Theta * Bt), where
    #   F(z) = BETA_MIN + (1 - BETA_MIN)/ln(2) * softplus(z)
    #   z = Bx^T * Theta * Bt
    # the unknown is Theta (Bx and Bt are fixed) => invert F(z) to obtain Theta
    w = (beta - BETA_MIN) * jnp.log(2) / (1 - BETA_MIN)
    z_exact = jnp.log(jnp.expm1(w))

    # least squares problem: 35 unknowns Theta_ij and NxNt data available
    #   z_exact = Bx^T * Theta * Bt
    # written using the pseudoinverse (?) as
    #   Bx * z_exact * Bt^T = (BxBx^T) * Theta * (BtBt^T)
    # split into 2 equations:
    #   1) (BxBx^T) * W = Bx * z_exact * Bt^T   => solve for W
    #   2) X = Theta * (BtBt^T), rewritten as 
    #       (BtBt^T)^T Theta^T = W^T
    #       BtBt^T Theta^T = W^T
    Bxx = B_x @ B_x.T
    Btt = B_t @ B_t.T
    W = jnp.linalg.solve(Bxx, B_x @ z_exact @ B_t.T)
    Theta = jnp.linalg.solve(Btt, W.T).T

    # calculate the min-square beta
    z_least_squares = B_x.T @ Theta @ B_t
    beta_fit = BETA_MIN + (1 - BETA_MIN) * jnp.logaddexp(0, z_least_squares) / jnp.log(2)

    err_beta = beta_fit - beta
    # global relative L2 error
    err_rel_L2 = jnp.linalg.norm(err_beta) / jnp.linalg.norm(beta)
    # pointwise relative error
    rel_err_beta = err_beta / beta
    # average absolute pointwise relative error
    avg_rel_err = jnp.mean(jnp.abs(rel_err_beta))

    aux.check_msg(f'Global relative L2 beta error       : {100*err_rel_L2:.2f}%')
    aux.check_msg(f'Max positive relative beta error    : {100*jnp.max(rel_err_beta):.2f}%')
    aux.check_msg(f'Max negative relative beta error    : {100*jnp.min(rel_err_beta):.2f}%')
    aux.check_msg(f'Max absolute relative beta error    : '
                  f'{100*jnp.max(jnp.abs(rel_err_beta)):.2f}%')
    aux.check_msg(f'Mean absolute relative beta error   : {100*avg_rel_err:.2f}%')