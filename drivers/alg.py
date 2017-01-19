import numpy as np
import sys
sys.path.insert(0, '../core')
import core
import utils
from interval import Interval
DEBUG = False

def RQI_step(D, U, H, mu, x, counter=None, inertia_only=False):
    ''' Do one step of RQI
    Params:
    mu[float]: Current eigenvalue guess, also the shift to use;
    x[np.ndarray]: Current eigenvector guess, with unit norm;
    
    Returns:
    mu[float]: Next eigenvalue guess;
    y[np.ndarray]: Next eigenvector guess;
    error[float]: Relative error in this step.
    n_mu[int]: number of eigenvalues smaller than mu
    '''
    try:
        D_hat, G = core.ldl(D-mu, U, H)
        y = core.lin_solve(D_hat, G, U, x)
        n_mu = utils.inertia_ldl(D_hat)[2]
    except ValueError as e:
        if counter:
            counter.stable_count += 1.0

        #print("Turning to stable linear solve method.")
        y, ratio, D_hat = core.SSQR_inertia(D-mu, U, H, x)
        n_mu = utils.inertia_qr(ratio, D_hat)[2]
    if inertia_only:
        return n_mu

    lam = np.dot(x, y)
    norm = np.linalg.norm(y)

    new_mu = mu + lam/norm**2
    #error = np.linalg.norm(x - lam/norm**2 *y)
    y = y / norm
    error = utils.comp_error(D, U, H, new_mu, y)
    return (new_mu, y, error, n_mu)

def inverse_iter(D, U, H, mu, x, eps = 1e-6):
    if DEBUG:
        print("Doing inverse iteration...")
    while True:
        v = x / np.linalg.norm(x)
        x, _, _ = core.SSQR(D-mu, U, H, v)
        theta = np.dot(x, v)
        if np.linalg.norm(x-theta*v) <= eps*abs(theta):
            break
        else:
            if DEBUG:
                print np.linalg.norm(x-theta*v)/abs(theta)
    mu += 1/theta
    x = x / theta
    error = utils.comp_error(D, U, H, mu, x)
    if DEBUG:
        print("Final error = %e." % (error))
    return mu, x, error

def RQI_fast(D, U, H, mu, x, counter=None, eps=1e-7, inertia_only=False):
    if counter is not None:
        counter.rqi_count += 1
    try:
        result = core.ldl_fast(D-mu, U, H)
        if len(result) == len(D):
            # Only D_hat is returned, meaning eigen-pair found
            n_mu = utils.inertia_ldl(result)[2]
            if inertia_only:
                return n_mu

            error = utils.comp_error(D, U, H, mu, x)
            #raise ValueError("Eval converged but error = %e." % error)
            '''Use inverse iteration to find eigenvector when eigenvalue is found
            Instead, lin_solve can be used to find eigenvector.'''
            if error > 1e-12:
                counter.inverse_count += 1
                mu, x, error = inverse_iter(D, U, H, mu, x, eps)
            
            return (mu, x, error, n_mu)
        else:
            D_hat, G = result
            y = core.lin_solve(D_hat, G, U, x)
            n_mu = utils.inertia_ldl(D_hat)[2]
    except ValueError as e:
        if counter:
            counter.stable_count += 1.0
        y, ratio, D_hat = core.SSQR_inertia(D-mu, U, H, x)
        n_mu = utils.inertia_qr(ratio, D_hat)[2]

    if inertia_only:
        return n_mu

    lam = np.dot(x, y)
    norm = np.linalg.norm(y)

    new_mu = mu + lam / norm**2
    y = y / norm
    error = utils.comp_error(D, U, H, new_mu, y)
    return (new_mu, y, error, n_mu)


def bisection_step(D, U, H, high, low, x, solve=True, counter=None):
    mu = (high+low)/2
    try:
        D_hat, G = core.ldl(D-mu, U, H)
        n_mu = utils.inertia_ldl(D_hat)[2]
        if solve:
            y = core.lin_solve(D_hat, G, U, x)
        else:
            y = x
    except ValueError as e:
        if counter:
            counter.stable_count += 1.0
        y, ratio, D_hat = core.SSQR_inertia(D-mu, U, H, x)
        n_mu = utils.inertia_qr(ratio, D_hat)[2]
        
    return (mu, y/np.linalg.norm(y), n_mu)

def initialize(D, U, H):
    n = len(D)
    x = np.random.randn(n)
    x = x /np.linalg.norm(x)
    low, high = min(D) - np.linalg.norm(H, 2), max(D) + np.linalg.norm(H, 2)
    _, _, _, n_high = RQI_step(D, U, H, high, x)
    _, _, _, n_low = RQI_step(D, U, H, low, x)
    mu = (high + low) / 2
    return (Interval(low, n_low, high, n_high), mu, x)