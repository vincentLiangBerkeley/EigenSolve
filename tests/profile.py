'''This file contains profiling stats for core subroutines as well as 
main algorithms and the driver function.'''
from context import core, utils, driver, alg
from utilities import random_example
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
def time_func(trials, func, *args):
	# func(D, U, H, b) should return x such that Ax = b
	ave = 0.0
	for it in xrange(trials):
		start = time.time()
		x = func(*args)

		ave += time.time() - start
	return ave / trials

def slope(x, y):
	# Given points x and y, find the slop of the best fit line
	# min||y-ax-b||
	A = np.vstack([x, np.ones(len(x))]).T
	m, c = np.linalg.lstsq(A, y)[0]
	return m, c

def slope(x, y):
	# Given points x and y, find the slop of the best fit line
	# min||y-ax-b||
	A = np.vstack([x, np.ones(len(x))]).T
	m, c = np.linalg.lstsq(A, y)[0]
	return m, c

def comp_inertia(trials=10):
	def inertia_fast(D, U, H):
		D_hat, G = core.ldl(D, U, H)
		inertia = utils.inertia_ldl(D_hat, G)
		return inertia

	def inertia_stable(D, U, H):
		x = np.ones(len(D))
		x, ratio, D_hat = core.SSQR_inertia(D, U, H, x)
		inertia = utils.inertia_qr(ratio, D_hat)
		return inertia

	qr_times, ldl_times = [], []
	ns = np.array([pow(2, i) for i in range(4, 11)])
	for n in ns:
		D, U, H = random_example(n, min(n//10, 10))
		print("\nTiming functions with n=%d, r=%d." % (n, min(n//10, 10)))
		ldl_times.append(time_func(trials, inertia_fast, D, U, H))
		qr_times.append(time_func(trials, inertia_stable, D, U, H))
	qr_times = np.array(qr_times)
	ldl_times = np.array(ldl_times)
	ave_ratio = np.mean(qr_times / ldl_times)
	slope_ldl, _ = slope(np.log2(ns), np.log2(ldl_times))
	slope_qr, _ = slope(np.log2(ns), np.log2(qr_times))

	plt.plot(np.log2(ns), np.log2(ldl_times), '-', color='r')
	plt.plot(np.log2(ns), np.log2(qr_times), color='blue')
	print("\nThe average time for QR to find inertia is %.4fx that for LDL." % ave_ratio)
	print("\nThe slop of th log-log plot for LDL and QR are %.4f and %.4f." % (slope_ldl, slope_qr))
	plt.show()

def time_driver():
	def eigen_solve(D, U, H):
		n = len(D)
		test_int, mu, x = alg.initialize(D, U, H)
		eintervals = [test_int]
		evals, evects = [], []
		for it in xrange(n):
			lamb, y = driver.find_pair(eintervals, D, U, H, mu, x, verbose=False)
			evals.append(lamb)
			evects.append(y)
			if it != n - 1:
				mu = (eintervals[0].low + eintervals[0].high) / 2
				x = np.random.randn(n)
				x = x/np.linalg.norm(x)
		return evals, evects
	eigen_times = []
	ns = np.array([pow(2, i) for i in range(4, 8)])
	for n in ns:
		D, U, H = random_example(n, min(n//10, 10))
		print("\nTiming eigen solver with n=%d, r = %d."%(n, min(n//10, 10)))
		eigen_times.append(time_func(1, eigen_solve, D, U, H))
	eigen_times = np.array(eigen_times)
	slop, _ = slope(np.log2(ns), np.log2(eigen_times))
	print("\nThe slope of log-log plot for eigen solve is %.4f." % slop)


if __name__ == "__main__":
	opt = sys.argv[1]
	if opt == 'inertia':
		print("Profiling two methods for computing inertia.")
		comp_inertia()
	elif opt == 'eigen':
		time_driver()
