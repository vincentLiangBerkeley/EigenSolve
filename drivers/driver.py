import sys
sys.path.insert(0, '../core')
import utils, core
import alg
from interval import Interval, find_interval
import output
import numpy as np
EPS_error = 1e-10
EPS_bisec = 1e-8

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
                    print("mu = %.8f, n_mid = %d, n_high=%d, n_low=%d." % (mu, n_mid, interval.n_high, interval.n_low))
                    print("Working on interval (%.4f, %.4f) with %d evals." % (interval.low, interval.high, interval.num_evals()))
                # Tests for convergence, needs to handle it better when converged
                if new_err < EPS_error:
                    result = core.ldl_fast(D-new_mu, U, H)
                    if len(result) == 2:
                        raise ValueError("new_mu = %e is not a real eigenvalue." % new_mu)
                    inertia = utils.inertia_ldl(result)
                    assert inertia[1] == 1
                    if verbose:
                        print("RQI Converged in: ")
                    if new_mu in interval:
                        if verbose:
                            output.display_interval(interval)
                        if interval.num_evals() > 1:
                            # Needs a better handling method for converging in interval with multiple eigenvalues
                            # Now see if new_mu is larger or smaller than mu because we know n_mid
                            if n_mid == interval.n_high:
                                boundary = mu - EPS_bisec
                                if version == 1:
                                    n_boundary = alg.RQI_step(D, U, H, boundary, x, counter, inertia_only=True)
                                else:
                                    n_boundary = alg.RQI_fast(D, U, H, boundary, x, counter, inertia_only=True)

                                result = Interval(interval.low, interval.n_low, mu, n_mid)
                                result.high = boundary
                                result.n_high -= max(abs(n_mid-n_boundary), 1)
                                eintervals.append(result)
                            else:
                                boundary = mu + EPS_bisec
                                if version == 1:
                                    n_boundary = alg.RQI_step(D, U, H, boundary, x, counter, inertia_only=True)
                                else:
                                    n_boundary = alg.RQI_fast(D, U, H, boundary, x, counter, inertia_only=True)

                                if type(n_boundary) != int:
                                    raise ValueError("n_boundary type error: %s" % type(n_boundary))
                                if abs(n_boundary - n_mid) > 1:
                                    raise ValueError("Multiple eigenvalues in interval: (%.8f, %.8f)"%(mu, boundary))
                                #print("n_boundary = %d, n_mid = %d." % (n_boundary, n_mid))
                                if n_mid == interval.n_low:
                                    result = Interval(mu, n_mid, interval.high, interval.n_high)
                                    result.low = boundary
                                    result.n_low += max(abs((n_boundary-n_mid)), 1)
                                    eintervals.append(result)
                                else:
                                    low_half, high_half = interval.split(mu, n_mid)
                                    high_half.low = boundary
                                    high_half.n_low += max(abs((n_boundary-n_mid)), 1)
                                    eintervals.append(low_half)
                                    if high_half.num_evals() > 0:
                                        eintervals.append(high_half)
                    else:
                        output.display_interval(interval)
                        raise ValueError("Eigenvalue %.8f does not converge in the interval ..." % new_mu)

                    return (new_mu, x)
                try:
                    eintervals += interval.split(mu, n_mid)
                except ValueError as e:
                    print e
                    n_low = alg.RQI_fast(D, U, H, interval.low, x, counter, inertia_only = True)
                    print("The real n_low = %d." % n_low)
                    raise ValueError
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
        # if verbose:
        #     key = raw_input("Press ENTER to continue...\n") 
        #     sys.stdout.flush()