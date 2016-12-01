from context import interval, alg, output
from utilities import normal_example
import numpy as np

class TestInterval:
	def test_interval(self, normal_example):
		n, r, D, U, H, A, b = normal_example
		test_int, mu, x = alg.initialize(D, U, H)
		evals, _ = np.linalg.eig(A)

		# Test basic initialization properties of test_int
		assert test_int.num_evals() == len(evals)
		assert test_int.low <= min(evals)
		assert test_int.high >= max(evals)

		# Test __contains__ method
		for e in evals:
			assert e in test_int

		lamb = np.percentile(evals, 50)
		n_mid = sum([e < lamb for e in evals])
		result = test_int.split(lamb, n_mid)
		# Test split method
		assert len(result) == 2
		i1, i2 = result
		assert i1.num_evals() == n_mid
		assert i2.num_evals() == test_int.n_high - n_mid

		# Do 2 steps of RQI to split the test_ints
		eintervals = [test_int]
		for i in range(3):
			new_mu, x, error, n_mid = alg.RQI_step(D, U, H, mu, x)
			it = interval.find_interval(mu, eintervals)
			assert it is not None
			output.display_interval(it)
			eintervals += it.split(mu, n_mid)
			output.display_intervals(eintervals)
			mu = new_mu