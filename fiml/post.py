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
import fiml.beta as betamod
from prettytable import PrettyTable



def check_experiments_in_sigma_band(sol, exp, ax_sol=2, N_sigma=2, t_plot=None, 
                                    fig_file=None, PLOT=True, VIDEO=False):
    # HACK: assicurati nel solver the la dimensione sia (Nt, Nx, N_sample),
    # non qui
    if sol.solution.ndim != 3:
        sol_values = sol.solution[jnp.newaxis, ...]
    else:
        sol_values = sol.solution
    # for each space-time point, calculate the sample mean of the solutions
    # obtained solving the equation for each sample
    # (1st index: space, 2nd index: time, 3rd index: sampling)
    u_mean = jnp.mean(sol_values, axis=ax_sol)
    u_sigma = jnp.std(sol_values, axis=ax_sol)
    u_upper = u_mean + N_sigma*u_sigma
    u_lower = u_mean - N_sigma*u_sigma
    
    # if no time instant to plot is given, the last one is assumed
    # (where discrepancy is likely to be the biggest)
    if t_plot is None:
        idx_plot_exp = -1
        aux.warning('No time instant given, the last instant '
                    f't = {exp.t[idx_plot_exp]} s will be plotted...')
    else:
        if t_plot < exp.t[0] or t_plot > exp.t[-1]:
            raise ValueError('Given time instant lies outside of the experimental range: \n'
                             f'Given time: {t_plot} s\n'
                             f'Experimental range: {exp.t[0]}-{exp.t[-1]} s')
        # find experimental point closest to the given time instant
        idx_plot_exp = jnp.argmin(jnp.abs(exp.t - t_plot))
    # interpolate numerical solution to the chosen experimental point 
    # (do NOT interpolate experimental data, as that could be meaningless!)
    t_exp_plot = exp.t[idx_plot_exp]
    u_mean_t = aux.sample_solution(u_mean, t_mesh=sol.t, t_probes=exp.t)
    u_up_t = aux.sample_solution(u_upper, t_mesh=sol.t, t_probes=exp.t)
    u_low_t = aux.sample_solution(u_lower, t_mesh=sol.t, t_probes=exp.t)
    
    # interpolate numerical data at probe locations to check whethet experimental
    # data lie withing +/- 2*sigma from the sample mean
    u_upper_probes = aux.sample_solution(u_upper, x_mesh=sol.x, t_mesh=sol.t, x_probes=exp.x, t_probes=exp.t)
    u_lower_probes = aux.sample_solution(u_lower, x_mesh=sol.x, t_mesh=sol.t, x_probes=exp.x, t_probes=exp.t)
    
    lies_within = (exp.solution <= u_upper_probes) & (exp.solution >= u_lower_probes)
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
    u_mean_probes = aux.sample_solution(u_mean_t[idx_plot_exp,:], x_mesh=sol.x, x_probes=exp.x)
    delta_u = u_mean_probes - exp.solution[idx_plot_exp,:]
    err = delta_u / exp.solution[idx_plot_exp,:]
    
    
    err_report = PrettyTable()
    err_report.title = f'Mean of the distribution vs Experimental data at t = {t_exp_plot:.5f} s'
    err_report.field_names = ['x', 'Mean', 'Experiment', 'Difference', 'Relative error [%]']
    for k in range(exp.solution.shape[1]):
        err_report.add_row([f'{exp.x[k]:.5f}', f'{u_mean_probes[k]:.5f}',
                            f'{exp.solution[idx_plot_exp,k]:.5f}', f'{delta_u[k]:.5f}', 
                            f'{100*err[k]:.2f}'])
    aux.check_msg(err_report.get_string())
    
    if PLOT:
        # data are stored as [t^k, u_1^k, ..., u_N^k] => create a column vector of
        # column vectors => flatten the 2D array in column-major order (Fortran-like)
        #sol_exp_t = jnp.ravel(sol_exp_interp_t.T)
        
        fig = plt.figure()
        plt.plot(sol.x, u_mean_t[idx_plot_exp,:], label='Sample mean', color='r')
        plt.fill_between(sol.x, u_up_t[idx_plot_exp,:], u_low_t[idx_plot_exp,:], alpha=0.3, color='r')
        #plt.plot(x, u_up_t[idx_plot_exp,:], color='r')
        #plt.plot(x, u_low_t[idx_plot_exp,:], color='r')
        plt.plot(exp.x, exp.solution[idx_plot_exp,:], '*', label='Experiment')
        plt.xlabel('$x$ [-]')
        plt.ylabel('$u$ [-]')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend()
        plt.title(fr'Sample mean solution with $\pm 2\sigma$ at $t = {t_exp_plot:.5f}$ s')
        plt.show(block=False)
        if fig_file is not None:
            aux.save_fig(fig, fig_file.name, fig_file.parent)


def compare_exact_beta(sol_objs, exact_sol, t_plot=None, fig_file=None, 
                       PLOT=True, VIDEO=False, beta_labels=('', )):
    
    STEADY = False
    if exact_sol.t.size == 1:
        aux.warning('Time mesh contains only one time instant, treating the '
                    'problem as steady...')
        STEADY = True
    # if no time instant to plot is given, the last one is assumed
    # (where discrepancy is likely to be the biggest)
    if not STEADY:
        if t_plot is None:
            idx_plot_beta = -1
            aux.warning('No time instant given, the last instant '
                        f't = {exact_sol.t[idx_plot_beta]} s will be plotted...')
        else:
            if t_plot < exact_sol.t[0] or t_plot > exact_sol.t[-1]:
                raise ValueError('Given time instant lies outside of the time range '
                                 'of the reference exact beta: \n'
                                 f'Given time: {t_plot} s\n'
                                 f'Experimental range: {exact_sol.t[0]}-{exact_sol.t[-1]} s')
            # find exact_beta point closest to the given time instant
            idx_plot_beta = jnp.argmin(jnp.abs(exact_sol.t - t_plot))
    else:
        # there's only one time instant: the last one
        idx_plot_beta = -1
    t_beta_plot = exact_sol.t[idx_plot_beta]
    
    # beta_num can either be
    #   - a single array, to check a single beta vs the exact one
    #   - a tuple of arrays (e.g. prior, posterior and MAP)
    if not isinstance(sol_objs, tuple):
        sol_objs = (sol_objs, )

    if not isinstance(beta_labels, tuple):
        beta_labels = tuple(beta_labels)
    if len(beta_labels) != len(sol_objs):
        raise ValueError('If labels are provided, each beta object must have a label!')
    
    # create figure out of the loop, then update it
    if PLOT:
        fig_beta, ax_beta = plt.subplots()
        ax_beta.plot(exact_sol.x, exact_sol.beta[:,idx_plot_beta], 'k', label=r'Exact $\beta$')
        ax_beta.set_xlabel(r'$x$ [-]')
        ax_beta.set_ylabel(r'$\beta$ [-]')
        ax_beta.autoscale(enable=True, axis='x', tight=True)
        time_str = f'at $t = {t_beta_plot:.5f}$ s' if not STEADY else ''
        ax_beta.set_title(fr'$\beta$ correction {time_str}')
        
        fig_err, ax_err = plt.subplots()
        ax_err.axhline(0, color='k', linestyle='--')
        ax_err.set_xlabel(r'$x$ [-]')
        ax_err.set_ylabel(r'$(\beta - \beta_\mathrm{ex})/\beta_\mathrm{ex}$ [\%]')
        ax_err.autoscale(enable=True, axis='x', tight=True)
        ax_err.set_title(fr'Errors (solid: $\beta$, dashed: $u$) {time_str}')
        ax_sol = ax_err.twinx()
        ax_sol.set_ylabel(r'$(u - u_\mathrm{ex})/u_\mathrm{ex}$ [\%]')
        ax_sol.tick_params(axis='y')
    
    min_err = jnp.inf
    max_err = -jnp.inf
    for i, sol_i in enumerate(sol_objs):
        # if a population of beta, result of the sampling, is passed (and thus
        # you have 3 dimensions: (Nx, Nt, N_samples)), average the beta's
        if sol_i.beta.ndim == 3:
            beta_i_mean = jnp.mean(sol_i.beta, axis=2)
            sol_i_mean = jnp.mean(sol_i.solution, axis=2)
        else:
            beta_i_mean = sol_i.beta
            sol_i_mean = sol_i.solution
    
        # numerical beta (from solver) reinterpolated over the space-time grid
        # of the exact beta (to avoid distorting the exact beta)
        # (pass the transpose because the function assumes shape (Nt, Nx))
        beta_xt = aux.sample_solution(beta_i_mean.T, x_mesh=sol_i.x, x_probes=exact_sol.x, 
                                      t_mesh=sol_i.t, t_probes=exact_sol.t)
        beta_xt = beta_xt.T
        
        sol_xt = aux.sample_solution(sol_i_mean, x_mesh=sol_i.x, x_probes=exact_sol.x,
                                     t_mesh=sol_i.t, t_probes=exact_sol.t)
        
        err_beta = (beta_xt - exact_sol.beta) / exact_sol.beta
        err_sol = (sol_xt - exact_sol.solution) / exact_sol.solution
        tmp_min = jnp.min(jnp.concatenate((jnp.ravel(err_beta), jnp.ravel(err_sol))))
        min_err = tmp_min if tmp_min < min_err else min_err
        tmp_max = jnp.max(jnp.concatenate((jnp.ravel(err_beta), jnp.ravel(err_sol))))
        max_err = tmp_max if tmp_max > max_err else max_err
        # average of the relative errors, in modulus
        MAE_err_t = jnp.mean(jnp.abs(err_beta[idx_plot_beta]))
        aux.check_msg(f'Beta object: {beta_labels[i]}\n'
                      f'At t = {t_beta_plot:.5f}:\n'
                      f'\tAverage of the relative errors, in modulus: {100*MAE_err_t:.2f}%\n'
                      f'\tMaximum error: {100*jnp.max(err_beta[idx_plot_beta]):.2f}%\n'
                      f'\tMinimum error: {100*jnp.min(err_beta[idx_plot_beta]):.2f}%')
    
        if PLOT:
            line, = ax_beta.plot(exact_sol.x, beta_xt[:,idx_plot_beta], 
                                 label=rf'{beta_labels[i]} $\beta$')
            line, = ax_err.plot(exact_sol.x, 100*err_beta[:,idx_plot_beta],
                                label=rf'{beta_labels[i]}', linestyle='-')
            # reuse previous line colour
            line, = ax_sol.plot(exact_sol.x, 100*err_sol[idx_plot_beta,:],
                                label=rf'{beta_labels[i]}', linestyle='--',
                                color=line.get_color())
            
        if VIDEO:
            fig, ax = plt.subplots()
            line_num, = ax.plot(exact_sol.x, jnp.zeros_like(exact_sol.x), label=r'Calculated $\beta$')
            line_exact, = ax.plot(exact_sol.x, jnp.zeros_like(exact_sol.x), label=r'Exact $\beta$')
            ax.legend(loc='upper right')
            ax.set_xlim(sol_i.x[0], sol_i.x[-1])
            ax.set_ylim(0, 2)
            # enable interactive mode
            plt.ion()
            plt.show(block=False)
            for j in range(sol_i.t.shape[0]):
                # stop animation if the user closed the window
                if not plt.fignum_exists(fig.number):
                    break
                line_num.set_ydata(beta_xt[:,j])
                line_exact.set_ydata(exact_sol.beta[:,j])
                fig.canvas.draw_idle()
                #fig.canvas.flush_events()
                #plt.pause(0.0000001)
                plt.pause(0.01)
                ax.set_title(f'$t = {exact_sol.t[j]:.4f}$ s')
            plt.ioff()
            # show figure only if it hasn't been closed by the user
            if plt.fignum_exists(fig.number):
                plt.show()

    if PLOT:
        beta_name = f'_{beta_labels[i]}'
        ax_beta.legend()
        fig_beta.show()
        if fig_file is not None:
            
            aux.save_fig(fig_beta, f'{fig_file.name}{beta_name}', fig_file.parent)
        
        # set same limits to have comparable error plots
        ax_err.set_ylim((100*min_err, 100*max_err))
        ax_sol.set_ylim((100*min_err, 100*max_err))
        fig_err.show()
        if fig_file is not None:
            aux.save_fig(fig_err, f'{fig_file.name}{beta_name}_error', fig_file.parent)
        
        

def compare_prior_posterior(prior, post=None, exp=None, MAP=None, ax_sol=2, N_sigma=2, t_plot=None, 
                            fig_file=None, PLOT=True, VIDEO=False):
    
    if post is not None:
        if ~jnp.all(jnp.isclose(prior.x, post.x)):
            raise ValueError('Spatial mesh of prior and posterior samples are different')
        if ~jnp.all(jnp.isclose(prior.t, post.t)):
            raise ValueError('Time mesh of prior and posterior samples are different')
    
    # HACK: assicurati nel solver the la dimensione sia (Nt, Nx, N_sample),
    # non qui
    if prior.solution.ndim != 3:
        prior_sol = prior.solution[jnp.newaxis, ...]
    else:
        prior_sol = prior.solution
    if post is not None:
        if post.solution.ndim != 3:
            posterior_sol = post.solution[jnp.newaxis, ...]
        else:
            posterior_sol = post.solution
    # for each space-time point, calculate the sample mean of the solutions
    # obtained solving the equation for each sample
    # (1st index: space, 2nd index: time, 3rd index: sampling)
    u_mean_prior = jnp.mean(prior_sol, axis=ax_sol)
    u_sigma_prior = jnp.std(prior_sol, axis=ax_sol)
    u_upper_prior = u_mean_prior + N_sigma*u_sigma_prior
    u_lower_prior = u_mean_prior - N_sigma*u_sigma_prior
    
    if post is not None:
        u_mean_post = jnp.mean(posterior_sol, axis=ax_sol)
        u_sigma_post = jnp.std(posterior_sol, axis=ax_sol)
        u_upper_post = u_mean_post + N_sigma*u_sigma_post
        u_lower_post = u_mean_post - N_sigma*u_sigma_post
    
    # if no time instant to plot is given, the last one is assumed
    # (where discrepancy is likely to be the biggest)
    if t_plot is None:
        idx_plot_exp = -1
        aux.warning('No time instant given, the last instant '
                    f't = {exp.t[idx_plot_exp]} s will be plotted...')
    else:
        if t_plot < exp.t[0] or t_plot > exp.t[-1]:
            raise ValueError('Given time instant lies outside of the experimental range: \n'
                             f'Given time: {t_plot} s\n'
                             f'Experimental range: {exp.t[0]}-{exp.t[-1]} s')
        # find experimental point closest to the given time instant
        idx_plot_exp = jnp.argmin(jnp.abs(exp.t - t_plot))
    # actual time instant plotted 
    t_exp_plot = exp.t[idx_plot_exp]
    # interpolate numerical solution at that time instant
    u_mean_prior_exp = aux.sample_solution(u_mean_prior, t_mesh=prior.t, t_probes=t_exp_plot)
    u_upper_prior_exp = aux.sample_solution(u_upper_prior, t_mesh=prior.t, t_probes=t_exp_plot)
    u_lower_prior_exp = aux.sample_solution(u_lower_prior, t_mesh=prior.t, t_probes=t_exp_plot)
    if post is not None:
        u_mean_post_exp = aux.sample_solution(u_mean_post, t_mesh=post.t, t_probes=t_exp_plot)
        u_upper_post_exp = aux.sample_solution(u_upper_post, t_mesh=post.t, t_probes=t_exp_plot)
        u_lower_post_exp = aux.sample_solution(u_lower_post, t_mesh=post.t, t_probes=t_exp_plot)
    
    if PLOT:
        fig = plt.figure()
        plt.plot(prior.x, u_mean_prior_exp, label='Prior', color='r')
        plt.fill_between(prior.x, u_upper_prior_exp, u_lower_prior_exp, alpha=0.3, color='r')
        
        if post is not None:
            plt.plot(post.x, u_mean_post_exp, label='Posterior', color='b')
            plt.fill_between(post.x, u_upper_post_exp, u_lower_post_exp, alpha=0.3, color='b')
        
        if MAP is not None:
            u_MAP = aux.sample_solution(MAP.solution, t_mesh=MAP.t, t_probes=t_exp_plot)
            plt.plot(MAP.x, u_MAP, label='MAP', color='tab:orange')
        
        plt.plot(exp.x, exp.solution[idx_plot_exp,:], 'k*', label='Experiment')
        plt.xlabel('$x$ [-]')
        plt.ylabel('$u$ [-]')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend()
        plt.title(fr'Prior vs Posterior at $t = {t_exp_plot:.5f}$ s')
        plt.show(block=False)
        if fig_file is not None:
            aux.save_fig(fig, fig_file.name, fig_file.parent)


def check_parametrisation_fitness(sol_exact, opts, PLOT=True, fig_file=None):
    
    if sol_exact.t.size == 1:
        STEADY = True
        idx_time = -1
    else:
        STEADY = False
        aux.warning('Unsteady data, but not time instant provided: will plot the last one...')
        idx_time = -1
    # signature: beta(theta, u=None, x=None, t=None)
    beta_fun, beta_params = betamod.parametrise_beta(opts, sol_exact.x, 
                                                     sol_exact.t, STEADY)
    
    # check beta ordering: it should be (Nt, Nx)^T, i.e. (Nx, Nt)
    Nx = sol_exact.x.size
    Nt = sol_exact.t.size
    
    # z here is always obtained using a matrix-vector product, and therefore
    # has a LINEAR relationship
    match opts['model']:
        case 'BxBt_softplus' | 'RBF_Gauss_x_softplus' | 'RBF_Gauss_u_softplus' | 'RBF_Gauss_x_poly_softplus':
            # softplus models:
            #   beta = beta_min + (1-beta_min)*softplus(z)/ln(2)
            # shape: (Nx, Nt)
            z_exact = betamod._invert_softplus(sol_exact.beta, opts['beta_min'], opts['beta_ref'])
        case 'BxBt_exp':
            # exp models:
            #   beta = beta_ref * e^z
            # shape: (Nx, Nt)
            z_exact = jnp.log(sol_exact.beta / opts['beta_ref'])
        case 'BxBt' | 'RBF_Gauss_x' | 'RBF_Gauss_u' | 'RBF_Gauss_x_poly' | 'RBF_Gauss_uu':
            # shape: (Nx, Nt)
            z_exact = sol_exact.beta - opts['beta_ref']
        case _:
            raise ValueError(f"Unknown beta model '{opts['model']}'")
    
    match opts['model']:
        case 'BxBt' | 'BxBt_softplus' | 'BxBt_exp':
            # least squares problem: 35 unknowns Theta_ij and Nx*Nt data available
            #   z = Bx^T * Theta * Bt
            # written using the pseudoinverse (?) as
            #   Bx * z * Bt^T = (BxBx^T) * Theta * (BtBt^T)
            # split into 2 equations:
            #   1) (BxBx^T) * W = Bx * z * Bt^T   => solve for W
            #   2) X = Theta * (BtBt^T), rewritten as 
            #       (BtBt^T)^T Theta^T = W^T
            #       BtBt^T Theta^T = W^T
            #Bxx = beta_params['B_x'] @ beta_params['B_x'].T
            #Btt = beta_params['B_t'] @ beta_params['B_t'].T
            #W = jnp.linalg.solve(Bxx, beta_params['B_x'] @ z_exact @ beta_params['B_t'].T)
            #Theta = jnp.linalg.solve(Btt, W.T).T
            # without using pseudo inverse: w = Theta*Bt, shape (M, Nt)
            # => solve Bx^T * w = z, where 
            w = jnp.linalg.lstsq(beta_params['B_x'].T, z_exact)[0]
            # solve Theta*Bt = w, i.e. Bt^T*Theta^T = w^T
            Theta = jnp.linalg.lstsq(beta_params['B_t'].T, w.T)[0].T
            # rewrite the matrix as a vector of parameters, as required by
            # the beta(theta, ...) function
            # Theta is a column vector of row vector => unravel in row-major
            # order (C-style)
            theta = jnp.ravel(Theta, order='C')
        case 'RBF_Gauss_x' | 'RBF_Gauss_x_softplus' | 'RBF_Gauss_x_poly' | 'RBF_Gauss_x_poly_softplus':
            if not STEADY:
                raise ValueError('RBF_Gauss_x and RBF_Gauss_x_softplus are only '
                                 'meant for STEADY problems!')
            # z = Phi * theta, with Phi = (Nx, N_RBF)
            # the k-th row of z is the z vector for the k-th time instant
            # => transpose it to have it as (Nx, 1), rather than (1, Nx)
            theta = jnp.linalg.lstsq(beta_params['Phi'], z_exact)[0]
            # theta: (N_theta, 1) -> (N_theta, )
            theta = jnp.ravel(theta)
        case 'RBF_Gauss_u' | 'RBF_Gauss_u_softplus' | 'RBF_Gauss_uu':
            # shape: (Nx, N_RBF, N_t)
            N_RBF = beta_params['N_params']
            if opts['model'] == 'RBF_Gauss_uu':
                Phi = betamod._RBF_Gauss_uu_Phi(sol_exact.solution, 
                                                beta_params['u_c'], 
                                                opts['shape_factor'])
            else:
                Phi = betamod._RBF_Gauss_u_Phi(sol_exact.x, beta_params['x_c'], 
                                               sol_exact.solution, 
                                               opts['shape_factor'])
            # at each time instant, the N_RBF equations holds:
            #   Phi[:,:,k] * theta = z[:,k]
            # since theta is always the same, it can be rewritten by stacking
            # in columns all the equations
            #   [  Phi[:,:,1]  ]           [  z[:,1].T  ]
            #   [  Phi[:,:,2]  ] * theta = [  z[:,2].T  ]
            #         ...                  [     ...    ]
            #   [  Phi[:,:,Nt] ]           [  z[:,Nt].T ]
            # reorder the axes so that it has shape (Nx, Nt, N_RBF)
            Phi_eq = jnp.transpose(Phi, (0, 2, 1))
            # and regroup them along the 3rd axis
            Phi_eq = jnp.reshape(Phi_eq, (Nx*Nt, N_RBF))
            z_eq = jnp.ravel(z_exact, order='C')
            theta = jnp.linalg.lstsq(Phi_eq, z_eq)[0]
            # theta: (N_theta, 1) -> (N_theta, )
            theta = jnp.ravel(theta)
        case _:
            raise ValueError(f"Unknown beta model '{opts['model']}'")

    beta_fit = beta_fun(theta, u=sol_exact.solution, x=sol_exact.x, t=sol_exact.t)

    err_beta = beta_fit - sol_exact.beta
    err_rel = err_beta / sol_exact.beta
    # global relative L2 error
    err_rel_L2 = jnp.linalg.norm(err_beta) / jnp.linalg.norm(sol_exact.beta)
    # pointwise relative error
    rel_err_beta = err_beta / sol_exact.beta
    # average absolute pointwise relative error
    avg_rel_err = jnp.mean(jnp.abs(rel_err_beta))

    aux.check_msg(f'Global relative L2 beta error       : {100*err_rel_L2:.2f}%')
    aux.check_msg(f'Max positive relative beta error    : {100*jnp.max(rel_err_beta):.2f}%')
    aux.check_msg(f'Max negative relative beta error    : {100*jnp.min(rel_err_beta):.2f}%')
    aux.check_msg(f'Max absolute relative beta error    : '
                  f'{100*jnp.max(jnp.abs(rel_err_beta)):.2f}%')
    aux.check_msg(f'Mean absolute relative beta error   : {100*avg_rel_err:.2f}%')
    
    if STEADY:
        theta_report = PrettyTable()
        time_str = f"at t = {sol_exact.t[idx_time]} s" if not STEADY else ''
        theta_report.title = f'Theta coefficients {time_str}'
        theta_report.field_names = ['Index', 'Value']
        for i, theta_i in enumerate(theta):
            theta_report.add_row([i+1, f'{theta_i:+.5e}'])
        aux.check_msg(theta_report.get_string())
    
    if PLOT:
        fig = plt.figure()
        plt.plot(sol_exact.x, sol_exact.beta[:,idx_time], label='Exact')
        plt.plot(sol_exact.x, beta_fit[:,idx_time], label='Fitted')
        plt.xlabel('$x$ [-]')
        plt.ylabel(r'$\beta$ [-]')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend()
        plt.show(block=False)
        if fig_file is not None:
            aux.save_fig(fig, fig_file.name, fig_file.parent)
        
        fig_err = plt.figure()
        plt.plot(sol_exact.x, 100*err_rel[:,idx_time])
        plt.xlabel('$x$ [-]')
        plt.ylabel(r'$|(\beta - \beta_\mathrm{ex})/\beta_\mathrm{ex}|$ [\%]')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.show(block=False)
        if fig_file is not None:
            aux.save_fig(fig_err, f'{fig_file.name}_error', fig_file.parent)
    
    return theta


def plot_convergence_history(opt_obj, fig_file=None):
    # iter = 0 is the initial guess
    iter_ax = jnp.arange(0, opt_obj.nit+1, step=1)
    
    fig_grad = plt.figure()
    plt.semilogy(iter_ax, opt_obj.history['grad_norm'], label='Exact')
    plt.xlabel('Iteration')
    plt.ylabel(r'$||\nabla\mathcal{J}(\theta)||$')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.show(block=False)
    if fig_file is not None:
        aux.save_fig(fig_grad, fig_file.name, fig_file.parent)
    
    fig_cost = plt.figure()
    plt.plot(iter_ax, opt_obj.history['cost'], label='Exact')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\mathcal{J}(\theta)$')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.show(block=False)
    if fig_file is not None:
        aux.save_fig(fig_cost, f'{fig_file.name}_cost', fig_file.parent)


def check_exact_beta_PDE(theta_ex, run_solver, sol_exact, problem, PLOT=True, 
                         fig_file=None, t_plot=None):
    # solve the problem using the "exact" theta obtained by fitting the exact
    # beta using the given parametrisation => if the fitting is good,
    # beta(theta_ex) = beta_ex => the PDE should be the original exact PDE,
    # therefore the solution should be the exact one
    solution, max_residual, success = run_solver(theta_ex)
    
    if solution.shape != sol_exact.solution.shape:
        exact_sol_resamp = aux.sample_solution(sol_exact.solution, 
                                               x_mesh=sol_exact.x, t_mesh=sol_exact.t, 
                                               x_probes=problem.x_mesh, t_probes=problem.t_mesh)
    else:
        exact_sol_resamp = sol_exact.solution
    delta = solution - exact_sol_resamp
    avg_delta = jnp.mean(jnp.abs(delta))
    rel_err = delta / exact_sol_resamp
    norm_err = jnp.linalg.norm(delta) / jnp.linalg.norm(exact_sol_resamp)
    aux.check_msg('Difference between PDE solution (at theta = theta_fit from '
                  'exact beta) and experimental data:'
                  f'\n\tMax absolute difference                     : {delta.max():.5e}'
                  f'\n\tMin absolute difference                     : {delta.min():.5e}'
                  f'\n\tAverage absolute difference (at all times)  : {avg_delta:.5e}'
                  f'\n\tL2-norm relative difference                 : {norm_err:.5e}')
    
    if PLOT:        
        fig_delta, ax = plt.subplots()
        # specify the physical extent of the domain and where is the origin of
        # the y-axis
        # aspect='auto' stretches the pixels, making them non-square => not
        # relevant here, since the 2 dimensions are not compatible (time-space)
        im = ax.imshow(jnp.abs(delta), cmap='hot', extent=[sol_exact.x[0], sol_exact.x[-1], 
                                                           sol_exact.t[0], sol_exact.t[-1]],
                       origin='upper', aspect='auto')
        ax.set_xlabel('$t$ [-]')
        ax.set_ylabel('$x$ [-]')
        ax.set_title(r'Solution at $\theta = \theta_\mathrm{fit}$ vs experiment')
        col_bar = fig_delta.colorbar(im, ax=ax)
        col_bar.set_label(r'$|u_\mathrm{ex} - u(\theta_\mathrm{fit})|$ [-]')
        if fig_file is not None:
            # plt.imshow() doesn't return a Figure, but an Axes object
            aux.save_fig(fig_delta.figure, f'{fig_file.name}_delta', fig_file.parent)
        plt.show(block=False)
        
        if t_plot is None:
            idx_plot_t = -1
            time_str = ''
        else:
            raise NotImplementedError("Still haven't implemented t_plot != None case")
        
        fig_compare = plt.figure()
        plt.plot(sol_exact.x, sol_exact.solution[idx_plot_t,:], label='Exact')
        plt.plot(problem.x_mesh, solution[idx_plot_t,:], label=r'u(x; $\theta = \theta_\mathrm{fit})$')
        plt.xlabel('$x$ [-]')
        plt.ylabel('$u$ [-]')
        plt.title(r'Solution at $\theta = \theta_\mathrm{fit}$ vs experiment')
        plt.legend()
        plt.autoscale(enable=True, axis='x', tight=True)
        if fig_file is not None:
            aux.save_fig(fig_compare, f'{fig_file.name}_uSolution', fig_file.parent)
        plt.show(block=False)
        
        fig_error = plt.figure()
        plt.plot(problem.x_mesh, 100*rel_err[idx_plot_t,:])
        plt.xlabel('$x$ [-]')
        plt.ylabel(r'$(u_\mathrm{ex} - u(\theta_\mathrm{fit}))/u_\mathrm{ex}$ [\%]')
        plt.title(r'Relative error')
        plt.autoscale(enable=True, axis='x', tight=True)
        if fig_file is not None:
            aux.save_fig(fig_error, f'{fig_file.name}_error', fig_file.parent)
        plt.show(block=False)


def plot_original_dataset(data, name_dataset=None, fig_file=None):
    fig_train = plt.figure()
    for case_i in data:
        sol = case_i.FI.MAP.solution.ravel(order='C')
        beta = case_i.FI.MAP.beta.ravel(order='F')
        plt.scatter(sol, beta, label=case_i.case_id)
    plt.xlabel('$u$ [-]')
    plt.ylabel(r'$\beta$ [-]')
    if name_dataset is not None:
        plt.title(r'$\{u, \beta\}$' + f' pairs in dataset {name_dataset}')
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.legend()
    if fig_file is not None:
        aux.save_fig(fig_train, f'{fig_file.name}', fig_file.parent)
    plt.show(block=False)
    
    

def plot_prepared_dataset(data, name_dataset=None, fig_file=None):
    fig = plt.figure()
    plt.scatter(data.sol, data.beta)
    plt.xlabel('$u$ [-]')
    plt.ylabel(r'$\beta$ [-]')
    if name_dataset is not None:
        plt.title(r'$\{u, \beta\}$' + f' pairs in dataset {name_dataset}')
    plt.autoscale(enable=True, axis='both', tight=True)
    if fig_file is not None:
        aux.save_fig(fig, f'{fig_file.name}', fig_file.parent)
    plt.show(block=False)