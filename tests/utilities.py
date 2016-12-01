import numpy as np
import pytest


class Counter(object):
    def __init__(self, bisec_count=0, rqi_count=0, stable_count=0):
        self.bisec_count = bisec_count
        self.rqi_count = rqi_count
        self.stable_count = stable_count

def sanity_check(eintervals, m):
    return sum([interval.num_evals() for interval in eintervals]) == m

def random_example(n, r, orth=True):
    '''Generate random matrix D+UHU^T'''
    U = np.random.randn(n, r)
    if orth:
        q, R = np.linalg.qr(U)
        U = q
    H = np.random.randn(r, r)
    H = (H+H.T)/2
    D = np.random.randn(n)
    return (D, U, H)

@pytest.fixture(scope='class')
def normal_example(n=256, r=10):
	print "\nGenerating normal example..."
	D, U, H = random_example(n, r)
	A = form_A(D, U, H)
	b = np.random.randn(n)
	return (n, r, D, U, H, A, b)

def form_L(D_hat, G, U):
    '''Form Lower triangular L from D_hat, G, U for test purpose'''
    n = len(D_hat)
    L = np.eye(n)
    for i in range(n-1):
        L[i+1:, i] = np.dot(U[i+1:, :], G[i, :].T)
    
    return L

def form_A(D, U, H):
    return np.diag(D) + np.dot(U, np.dot(H, U.T))
