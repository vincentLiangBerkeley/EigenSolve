import sys
sys.path.insert(0, '../core')
import utils, core
import alg
from interval import Interval, find_interval
import output
import numpy as np
EPS_error = 1e-10
EPS_bisec = 1e-10

def bisection_fallback(eintervals, D, U, H, x, solve=True, counter=None):
    interval = eintervals.pop()
    print("Falling back to bisection step in interval (%.4f, %.4f)" % (interval.low, interval.high))
    print("Solve = %d." % solve)
    if interval.high - interval.low <= EPS_bisec:
        print('Interval too small, already converged, inverval containing %d evals.' % interval.num_evals())
        mu = (interval.low + interval.high) / 2
        
        return (mu, x)
    new_mu, x, n_mid = alg.bisection_step(D, U, H, interval.high, interval.low, x, solve, counter)
    # Add the bisected intervals to queue.
    eintervals += interval.split(new_mu, n_mid)

    if solve:
        old_err = utils.comp_error(D, U, H, new_mu, x)
        mu = (eintervals[-1].low + eintervals[-1].high) / 2
    else:
        mu = (eintervals[0].low + eintervals[0].high) / 2
        # Could use cleverer initialization method
        x = np.random.randn(len(D))
        x = x / np.linalg.norm(x)
        old_err = utils.comp_error(D, U, H, mu, x)
        
    return (mu, x, old_err)

def find_pair(eintervals, D, U, H, mu, x, verbose=True, counter=None, version=1):
    old_err = utils.comp_error(D, U, H, mu, x)
    while True:
        if verbose:
            output.display_intervals(eintervals)
            #output.plot_intervals(eintervals, mu)
        # Do one step of RQI to get next estimate
        
        if counter is not None:
            counter.rqi_count += 1.0
        if version == 1:
            new_mu, x, new_err, n_mid = alg.RQI_step(D, U, H, mu, x, counter)
        elif version == 2:
            new_mu, x, new_err, n_mid = alg.RQI_fast(D, U, H, mu, x, counter)
        ratio = old_err/new_err
        if verbose:
            print("Error decrease ratio = %.4f, new_error = %e" % (ratio, new_err))
        
        # See if the progress is large enough
        if ratio > 1.5:
            # Try to locate the last estimate
            interval = find_interval(mu, eintervals)
            if interval is not None:
                if verbose:
                    print("mu = %.8f, n_mid = %d." % (mu, n_mid))
                    print("Working on interval (%.4f, %.4f) with %d evals." % (interval.low, interval.high, interval.num_evals()))
                # Tests for convergence, needs to handle it better when converged
                if new_err < EPS_error:
                    if verbose:
                        print("RQI Converged in: ")
                    if new_mu in interval:
                        if verbose:
                            output.display_interval(interval)
                        if interval.num_evals() > 1:
                            # Needs a better handling method for converging in interval with multiple eigenvalues
                            # Now see if new_mu is larger or smaller than mu because we know n_mid
                            if n_mid != interval.n_low and n_mid != interval.n_high:
                                result = interval.split(mu, n_mid)
                            elif n_mid == interval.n_low:
                                result = Interval(mu, n_mid, interval.high, )

                    return (new_mu, x)
                eintervals += interval.split(mu, n_mid)
                # Update the next estimate
                mu = new_mu
                old_err = new_err
            else:
                if verbose:
                    print("mu=%.8f is currently outside, see how it goes..." % mu)
                if ratio > 1e5:
                    if verbose:
                        print("Almost converged, need to stop it and revert to bisection.")
                    # Needs to handle convergence by bisection here
                    if counter is not None: 
                        counter.bisec_count += 1.0
                    results = bisection_fallback(eintervals, D, U, H, x, solve=False, counter=counter)
                    if len(results) == 2:
                        mu, x = results
                        
                        print("Bisection converged mu = %.4f, doing inverse iteration..." % mu)
                        # Now do inverse iteration to find eigenvector
                        mu, x, error = alg.inverse_iter(D, U, H, mu, x)
                        print("Final error = %e." % error)
                        return mu, x
                    else:
                        mu, x, old_err = results
                else:
                    # Update the next estimate
                    mu = new_mu
                    old_err = new_err
        else:
            if counter is not None:
                counter.bisec_count += 1.0
            mu, x, old_err = bisection_fallback(eintervals, D, U, H, x, solve=True, counter=counter)
        if verbose:
            key = raw_input("Press ENTER to continue...\n") 
            sys.stdout.flush()