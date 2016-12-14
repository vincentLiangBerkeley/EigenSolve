from context import alg, utils, core
from utilities import normal_example
import numpy as np

class TestRQI:
	def test_rqi_step(self, normal_example):
		n, r, D, U, H, A, b = normal_example
		# Create a random first guess mu, and vector will be b
		mu = np.random.randn()
		new_mu, y, error, n_mid = alg.RQI_step(D, U, H, mu, b)
		x = np.linalg.solve(A - mu*np.eye(n), b / np.linalg.norm(b))
		assert np.allclose(x/np.linalg.norm(x), y)
		mu_i = np.dot(x, np.dot(A, x)) / np.dot(x, x)
		assert np.allclose(mu_i, new_mu) 
		evals, _ = np.linalg.eig(A)
		assert n_mid == sum([e < mu for e in evals])

	def test_rqi_converge(self, normal_example):
		n, r, D, U, H, A, b = normal_example
		b = b / np.linalg.norm(b)
		mu = np.dot(b.T, np.dot(A, b))
		for i in range(10):
			mu, b, error, n_mid = alg.RQI_step(D, U, H, mu, b)
			print error
			if error < 1e-10:
				print("Converged in %d iterations." % (i+1))
				break
		D_hat = core.ldl_fast(D-mu, U, H)
		inertia = utils.inertia_ldl(D_hat)
		print inertia
		assert np.allclose(np.dot(A, b), mu*b)

	def test_rqi_fast_converge(self, normal_example):
		# RQI_fast has a problem where the eigen vector has not converged
		n, r, D, U, H, A, b = normal_example
		b = b / np.linalg.norm(b)
		mu = np.dot(b.T, np.dot(A, b))
		old_mu = mu
		for i in range(20):
			mu, b, error, n_mid = alg.RQI_fast(D, U, H, mu, b)
			ratio = abs(old_mu - mu) / abs(old_mu)
			print "Relative error between guesses: %e, error = %e" % (ratio, error)
			old_mu = mu
			if ratio < 1e-8:
				print("Converged in %d iterations." %(i+1))
				break

		D_hat = core.ldl_fast(D-mu, U, H)
		inertia = utils.inertia_ldl(D_hat)
		print inertia

class TestBisec:
	def test_bisec_step(self, normal_example):
		n, r, D, U, H, A, b = normal_example
		start, end = min(D) - np.linalg.norm(H, 2), max(D) + np.linalg.norm(H, 2)
		mu, y, n_mid = alg.bisection_step(D, U, H, start, end, b, solve=True)
		assert mu == (start+end)/2
		x = np.linalg.solve(A - mu*np.eye(n), b)
		assert np.allclose(x/np.linalg.norm(x), y)
		evals, _ = np.linalg.eig(A)
		assert n_mid == sum([e < mu for e in evals])

	def test_bisec_converge(self, normal_example):
		n, r, D, U, H, A, b = normal_example
		low, high = min(D) - np.linalg.norm(H, 2), max(D) + np.linalg.norm(H, 2)
		_, _, _, n_high = alg.RQI_step(D, U, H, high, b)
		_, _, _, n_low = alg.RQI_step(D, U, H, low, b)
		i = 0
		x = b / np.linalg.norm(b)
		mid = (high + low)/2
		while (high - low > 1e-10):
			i += 1
			mid, x, n_mid = alg.bisection_step(D, U, H, high, low, x)
			# Search for the upper half interval
			if n_high - n_mid > 0:
				n_low = n_mid
				low = mid
			else:
				n_high = n_mid
				high = mid
		print("Converged in %d steps with mu=%.4f." % (i, mid))
		assert np.allclose(np.dot(A, x), mid*x)

	def test_eig(self, normal_example):
		n, r, D, U, H, A, b = normal_example
		print("n = %d" % n)
		evals, evecs = np.linalg.eig(A)