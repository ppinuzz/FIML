"""
Utilities to manage parallel execution of the code.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import fiml.auxiliaries as aux
from concurrent.futures import ProcessPoolExecutor, as_completed
import cloudpickle
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from time import perf_counter

# global variable => each worker has its own WORKER_INPUTS variabile, which is
# global to that worker, but it's NOT shared with other workers
WORKER_ARGS = None

def run_parallel(N_workers, theta, run_solver_case, solution, residuals, success,
                 *solver_args):
    start_time = perf_counter()
    # serialise the quantities that are not changing per each run
    # (i.e. everything besided the r-th draw of theta and the run ID 'r')
    # (serialise = transform into a sequence of bytes)
    # problem: serialising and de-serialising the same function produces two
    # different objects => solver recompiled each time
    serialised_args = cloudpickle.dumps((run_solver_case, solver_args))
    # 1) when the workers pool is created, _initialise_worker() is executed only
    # once per each worker at the beginning
    #   - 'initiargs' contains the input arguments of the initialiser functions
    # 2) _initialise_worker() loads the pickled objects and store them in the
    # global variable, so that all the jobs handled by the i-th worker have
    # access to those => same function objects beta and nu (which, if parametrised,
    # depend solely on the array of parameters) => no need to recompile anything
    with ProcessPoolExecutor(max_workers=N_workers, initializer=_initialise_worker,
                             initargs=(serialised_args, )) as executor:
        job_handles = {}
        for r, theta_r in enumerate(theta):
            print(f'Submitting job {r}...')
            # submit a job to the process pool
            # submit() doesn't wait for the job to finish, but immediately
            # returns a Future object (a handle to the submitted job) which
            # contains data as job status, job results, raised exceptions, ecc.
            job = executor.submit(_run_worker, r, theta_r)
            # job_future.result() returns the output of the solver run 
            # retrievable using the 'job_future' handle
            # store the association between the job handle and the sample
            # number
            job_handles[job] = r
        print('All jobs have been submitted')
        # each run can take a different amount of time => as soon as each run
        # finishes
        for job in as_completed(job_handles):
            # get the sample number associated to the finished jobs
            sample_num = job_handles[job]
            # the solver might return an Exception (rather than crashing) =>
            # the raised exception is stored in the Future job object and gets
            # re-raised when .result() is called
            try:
                # residuals' 2nd dimension 1 is not really needed, but that way 
                # they have the same calling scheme (time, space, sample) as
                # the solution object
                _, sol, res, status = job.result()
                solution = solution.at[:,:,sample_num].set(jnp.asarray(sol))
                residuals = residuals.at[:,0,sample_num].set(jnp.asarray(res))
                success = success.at[sample_num].set(jnp.asarray(status))
            except Exception as exc:
                # sample_id is not available here because .result() crashed
                # => use sample_num, which should be equal
                aux.warning(f'Job {sample_num} failed with error message: {exc.args[0]}')
            else:
                if status:
                    print(f'Job {sample_num} completed successfully')
                else:
                    aux.warning(f'Job {sample_num} completed but did not converge/contains NaNs')
                
    elapsed_time = perf_counter() - start_time
    print(f'Total parallel execution time: {elapsed_time:.3f} s')
    
    return solution, residuals, success


def _initialise_worker(serialised_args):
    global WORKER_ARGS
    # deserialise the solver arguments which are not changing between calls,
    # storing the output in a per-worker global variable
    WORKER_ARGS = cloudpickle.loads(serialised_args)


def _run_worker(r, theta_r):
    run_solver_case, solver_args = WORKER_ARGS
    run_id, sol, res, status = run_solver_case(r, theta_r, *solver_args)
    # JAX can execute things asynchronously => time calculated with perf_counter()
    # may not be the actual time, as JAX could still be running when the 2nd
    # perf_counter() is called => force JAX to finish and return control to 
    # the Python process
    sol, res, status = jax.device_get((sol, res, status))
    return run_id, sol, res, status