"""
Utilities to manage parallel execution of the code.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

import fiml.auxiliaries as aux
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_parallel(N_workers, theta, run_solver_case, *solver_args):
    with ProcessPoolExecutor(max_workers=N_workers) as executor:
        job_handles = {}
        for r, theta_r in enumerate(theta):
            print(f'Submitting job {r}...')
            # submit a job to the process pool
            # submit() doesn't wait for the job to finish, but immediately
            # returns a Future object (a handle to the submitted job) which
            # contains data as job status, job results, raised exceptions, ecc.
            # all the 
            job = executor.submit(run_solver_case, r, theta_r, *solver_args)
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
                # (sample_id should be equal to sample_num)
                # output will be a list of all the values returned by the solver
                sample_id, *output = job.result()
            except Exception as exc:
                # sample_id is not available here because .result() crashed
                # => use sample_num, which should be equal
                aux.warning(f'Job {sample_num} failed with error message: {exc.args[0]}')
            else:
                print(f'Job {sample_id} completed successfully')
    
    return job_handles

