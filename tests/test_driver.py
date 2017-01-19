from context import driver_back, alg, utils, interval, output, core
import numpy as np
from utilities import random_example, form_A, Counter, sanity_check
import pytest
import time

@pytest.fixture(scope='class')
def normal_example(n=256, r=10):
	print "\nGenerating normal example..."
	D, U, H = random_example(n, r)
	A = form_A(D, U, H)
	b = np.random.randn(n)
	return (n, r, D, U, H, A, b)

def run_driver(v, D, U, H, verbose=False):
		# Generate small examples to walk through
		test_int, mu, x = alg.initialize(D, U, H)
		eintervals = [test_int]

		evals, evects = [], []
		
		RQI_iters, bisec_iters, stable, inverse, solve_count = [], [], [], [], []
		n = len(D)
		for it in xrange(n):
			#print it, sum([interval.num_evals() for interval in eintervals])
			assert sanity_check(eintervals, n-it)
			#print("Finding the %dth eigen pair.\n" % (it+1))
			counter = Counter()
			lamb, y = driver_back.find_pair(eintervals, D, U, H, mu, x, counter=counter, version = v)
			evals.append(lamb)
			evects.append(y)

			# D_hat = core.ldl_fast(D-lamb, U, H)
			# inertia = utils.inertia_ldl(D_hat)
			# assert inertia[1] == 1
			#assert np.allclose(0, utils.comp_error(D, U, H, lamb, y) )

			RQI_iters.append(counter.rqi_count)
			bisec_iters.append(counter.bisec_count)
			stable.append(counter.stable_count)
			inverse.append(counter.inverse_count)
			solve_count.append(counter.solve_count)

			if it != n - 1:
				mu = (eintervals[0].low + eintervals[0].high) / 2
				# Can initialize smarter
				x = np.random.randn(n)
				x = x / np.linalg.norm(x)
			
		
		RQI_iters = np.array(RQI_iters)
		bisec_iters = np.array(bisec_iters)
		stable = np.array(stable)
		print("Max number of RQI iterations = %d, that of bisection = %d." % (max(RQI_iters), max(bisec_iters)))
		
		print("Tol number of RQI iterations = %d, that of bisection = %d." % (sum(RQI_iters), sum(bisec_iters)))
		print("Ave number of RQI iterations = %.2f, that of bisection = %.2f." % (np.mean(RQI_iters), np.mean(bisec_iters)))
		print("Tol number of inverse iterations = %d." % sum(inverse))
		print("The total number of stable QR method is %d." % (sum(stable)))
		print("The total number of bisection that need solve is %d." % sum(solve_count))

		A = form_A(D, U, H)
		for i in xrange(n):
			assert np.allclose(np.dot(A, evects[i]), evals[i]*evects[i])

class TestDriver:
	def test_driver(self, normal_example):
		n, r, D, U, H, _, _ = normal_example
		# print("Testing driver with RQI version 1...")
		# start = time.time()
		# run_driver(1, D, U, H, verbose=False)
		# v1 = time.time() - start
		print("\nTesting driver with RQI version 2...")
		start = time.time()
		run_driver(2, D, U, H, verbose=False)
		v2 = time.time() - start
		#print("On n = %d, the second version is %.4f times faster than the first." % (n, v1 / v2))
	def test_interval_split(self, normal_example):
		n, r, D, U, H, A, b = normal_example
		# Guess a close value to a true eigenvalue and see how interval split is performed
		evals, evects = np.linalg.eig(A)
		index = np.random.choice(n)
		mu = evals[index]
		x = evects[index]
		eintervals = [interval.Interval(min(evals)-1, 0, max(evals)+1, n)]
		print("Running one step of eigen solve...")
		lamb, y = driver.find_pair(eintervals,D, U, H, mu, x, version=1)
		output.display_intervals(eintervals)