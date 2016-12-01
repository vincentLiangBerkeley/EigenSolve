from context import core, utils
from utilities import normal_example, form_L
import numpy as np
import pytest

class TestLDL:
	def test_ldl_normal(self, normal_example):
		n, r, D, U, H, A, b = normal_example
		D_hat, G = core.ldl(D, U, H)
		L = form_L(D_hat, G, U)
		assert np.allclose(A, np.dot(L, np.dot(np.diag(D_hat), L.T)))	

	def test_ldl_except(self, normal_example):
		n, r, D, U, H, A, b = normal_example
		#with pytest.raises(ValueError, message="D_hat is too small, matrix is singular!"):
		evals, evecs = np.linalg.eig(A)
		index = np.random.randint(n)
		print("Subtracting the %dth eigenvalue." % (index+1))
		D_hat = D - evals[index]
		A_hat = A - np.eye(n) * evals[index]
		result = core.ldl(D_hat, U, H)
		assert len(result) == n
		
	def test_lin_solve(self, normal_example):
		n, r, D, U, H, A, b = normal_example
		D_hat, G = core.ldl(D, U, H)
		x = core.lin_solve(D_hat, G, U, b)
		x_hat = np.linalg.solve(A, b)
		assert np.allclose(x, x_hat)

	def test_ldl_inertia(self, normal_example):
		n, r, D, U, H, A, b = normal_example
		H = np.diag(np.random.randn(r))
		D = np.random.rand(n) * (-1 * 5.0)
		A = np.diag(D) + np.dot(U, np.dot(H, U.T))
		evals, evecs = np.linalg.eig(A)
		D_hat, G = core.ldl(D, U, H)
		(pos, zero, neg) = utils.inertia_ldl(D_hat, G)
		assert pos == sum([e > 0 for e in evals])
		assert neg == sum([e < 0 for e in evals])

@pytest.fixture(scope='class')
def unstable_example(normal_example):
	print "\nGenerating unstable example..."
	n, r, D, U, H, A, b = normal_example
	evals, evecs = np.linalg.eig(A)
	index = np.random.randint(len(evals))
	D_hat = D - evals[index]
	A_hat = A - np.eye(n) * evals[index]
	x_hat = np.random.randn(n)
	b_hat = np.dot(A_hat, x_hat)
	return (A_hat, D_hat, U, H, b_hat, x_hat, evals, index)

class TestQR:
	def test_ssqr_normal(self, normal_example):
		n, r, D, U, H, A, b = normal_example
		x, V, HUT = core.SSQR(D, U, H, b)
		assert np.allclose(np.dot(A, x), b)

	def test_inverse_iter(self, normal_example):
		n, r, D, U, H, A, b = normal_example
		evals, _ = np.linalg.eig(A)
		index = np.random.randint(n)
		mu = evals[index]

		x = b / np.linalg.norm(b)
		err = utils.comp_error(D, U, H, mu, x)
		while err > 1e-9:
			x, _, _ = core.SSQR(D-mu, U, H, x)
			x = x / np.linalg.norm(x)
			err = utils.comp_error(D, U, H, mu, x)
			print err

	def test_ssqr_inertia(self, normal_example):
		n, r, D, U, H, A, b = normal_example
		evals, evecs = np.linalg.eig(A)
		x, ratio, D_hat = core.SSQR_inertia(D, U, H, b)
		assert np.allclose(np.dot(A, x), b)
		inertia = utils.inertia_qr(ratio, D_hat)
		assert inertia[0] == sum([e > 0 for e in evals])
		assert inertia[2] == sum([e < 0 for e in evals])
		assert inertia[2] > 0

	def test_ssqr_unstable(self, unstable_example):
		A_hat, D_hat, U, H, b_hat, x_hat, evals, index = unstable_example
		x, V, HUT = core.SSQR(D_hat, U, H, b_hat)
		assert np.allclose(np.dot(A_hat, x), b_hat)
		n = len(D_hat)
		x, V, HUT = core.SSQR(D_hat, U, H, np.zeros(n))
		assert np.allclose(np.dot(A_hat, x), np.zeros(n))

	def test_ssqr_inertia_unstable(self, unstable_example):
		A_hat, D_hat, U, H, b_hat, x_hat, evals, index = unstable_example
		x, ratio, d_hat = core.SSQR_inertia(D_hat, U, H, b_hat)
		assert np.allclose(np.dot(A_hat, x), b_hat)
		in_qr = utils.inertia_qr(ratio, d_hat)
		result = core.ldl(D_hat, U, H)
		assert len(result) == len(D_hat)
		in_ldl = utils.inertia_ldl(result)
		print in_qr, in_ldl