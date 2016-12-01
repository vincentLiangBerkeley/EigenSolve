import numpy as np
import utils
'''Module of all core subroutines
This is a module containing all core subroutines for doing LDL decomposition and stable QR solve
on a matrix of the form D + UHU^T where D is diagonal and H is small and symmetric.
TODO: Optimize subroutines in here
'''

def ldl_fast(D, U, H, eps=1e-11):
	'''Do LDL decomposition of a symmetric matrix
	If the matrix happens to be singular in the middle, an ValueError will be raised.

	Params:
	-------
	D : List or ndarray
		The diagonal entries of the D
	U : numpy.ndarray
		The orthogonal matrix
	H : numpy.ndarray
		r by r symmetric matrix

	Returns:
	--------
	D_hat : numpy.ndarray
		diagonal entries in LDL decomposition
	G : numpy.ndarray
		A matrix storing information such that L = G*U
	'''
	n, r = U.shape
	G = np.zeros((n, r))
	D_hat = np.zeros(n)

	for i in range(n):
		g = H.dot(U[i, :].T)
		D_hat[i] = D[i] + np.dot(U[i, :], g)
		if abs(D_hat[i]) < eps:
			# Have to change so that if i == n-1, need to return instead of raising error
			if i == n - 1:
				return D_hat
			else:
				raise ValueError('%d' % (i+1))
		G[i, :] = g.T / D_hat[i]
		H = H - np.outer(g, G[i, :])
	return (D_hat, G)

def ldl(D, U, H, eps=1e-11):
	n, r = U.shape
	G = np.zeros((n, r))
	D_hat = np.zeros(n)

	for i in range(n):
		g = H.dot(U[i, :].T)
		D_hat[i] = D[i] + np.dot(U[i, :], g)
		if abs(D_hat[i]) < eps:
			# Have to change so that if i == n-1, need to return instead of raising error
			raise ValueError('%d' % (i+1))
		G[i, :] = g.T / D_hat[i]
		H = H - np.outer(g, G[i, :])
	return (D_hat, G)

def SSQR(D, U, H, b):
    # Might try to redesign to only return the QR factors
	n, r = np.shape(U)
	HUT = np.dot(H, U.T)
    # Storing a portion of Q of the QR factor
	G = np.eye(r)
	# Storing the diagonal entries for the R factor
	R_diag = np.zeros(n)
	W = np.vstack((-1*np.eye(r), U))
	# V*H*U^T will be the non-diagonal part of the R factor, stored in V row by row
	V = np.zeros((n, r))
	b_hat = np.concatenate((np.zeros(r), b))
	
	for k in range(n):
	    # Do n steps of QR decomposition on the first n columns
	    
	    # Take the first column, only the non-zero entries
		g1 = np.concatenate((np.dot(G, HUT[:,k]), np.array([D[k]])))
		c, h = utils.house(g1)
		R_diag[k] = c
		g2 = np.dot(h.T, W[k:k+r+1, :])
		W[k:k+r+1, :] -= 2*np.outer(h, g2)
		
		g3 = np.dot(h.T, b_hat[k:k+r+1])
		b_hat[k:k+r+1] -= 2*g3*h
		    
		g4 = np.dot(h[:-1].T, G)
		V[k, :] = G[0, :] - 2*h[0]*g4
   		G = np.vstack((G[1:, :], np.zeros((1, r)))) - 2*np.outer(h[1:], g4)
    
	Q, R = np.linalg.qr(W[n:, :])
	b_hat[n:] = np.dot(Q.T, b_hat[n:])
	b_hat[n:] = np.linalg.solve(R, b_hat[n:])
	x = b_hat[0:n] - np.dot(W[0:n, :], b_hat[n:])
	tau = np.zeros(r)
	
	for i in range(n-1, -1, -1):
	    x[i] = (x[i] - np.dot(V[i, :], tau)) / R_diag[i]
	    tau += x[i] * HUT[:, i]
	return (x, V, HUT)

def SSQR_inertia(D, U, H, b):
    # A = D = UHU^T
    # Solve for Ax=b and compute inertia of A at the same time
    # Essentially doing QR decomposition and solving for Ax=e_i at every iteration
	n, r = U.shape
	HUT = np.dot(H, U.T)
	G = np.eye(r)
	W = np.vstack((-1*np.eye(r), U))
	GW = np.eye(r+1)
	D_hat = np.zeros(n)
	V = np.zeros((n, r))
	b_hat = np.concatenate((np.zeros(r), b))
	inertia = np.zeros(n)
	
	for k in xrange(n):
		for t in range(r):
			g1 = np.array([W[k+t, t], W[k+r, t]])
			Giv, y = utils.planerot(g1)
			W[[k+t, k+r], t:r] = np.dot(Giv, W[[k+t, k+r], t:r])
			GW[:, [t, r]] = np.dot(GW[:, [t, r]], Giv.T)
		g1 = np.concatenate((np.dot(G, HUT[:, k]), np.array([D[k]])))
		c, h = utils.house(g1)
		D_hat[k] = c
		
		z = (-2*h[r])*h
		z[r] += 1
		g2 = np.dot(h.T, GW)
		GW -= 2*np.outer(h, g2)
		for t in range(r-1, -1, -1):
			Giv, y = utils.planerot(GW[0, [t,t+1]])
			GW[:, [t,t+1]] = np.dot(GW[:, [t,t+1]], Giv.T)
			W[[k+t, k+t+1], t:r] = np.dot(Giv, W[[k+t, k+t+1], t:r])
		W[k, :] *= GW[0, 0]
		z[1:r+1] = np.dot(GW[1:r+1, 1:r+1].T, z[1:r+1])
		R = np.triu(W[k+1:k+r+1, :])
		z[1:r+1] = np.linalg.solve(R, z[1:r+1])
		inertia[k] = z[0] - np.inner(W[k, :], z[1:r+1])
		GW[0:r, 0:r] = GW[1:r+1, 1:r+1]
		GW[0:r, r], GW[r, 0:r], GW[r, r] = 0, 0, 1
		
		gamma2 = np.dot(h.T, b_hat[k:k+r+1])
		b_hat[k:k+r+1] -= 2*h*gamma2
		
		g3 = np.dot(h[0:r].T, G)
		V[k, :] = G[0, :] - 2*h[0]*g3
		G = np.vstack((G[1:, :], np.zeros((1, r)))) - 2*np.outer(h[1:], g3)
	
	b_hat[n:] = np.dot(GW[0:r, 0:r].T, b_hat[n:])
	R = np.triu(W[n:, :])
	b_hat[n:] = np.linalg.solve(R, b_hat[n:])
	x = b_hat[0:n] - np.dot(W[0:n, :], b_hat[n:])
	tau = np.zeros(r)
	for i in range(n-1, -1, -1):
	    x[i] = (x[i] - np.dot(V[i, :], tau)) / D_hat[i]
	    tau += x[i] * HUT[:, i]
	return (x, inertia, D_hat)

def lin_solve(D_hat, G, U, b):
	'''
	Assume D_hat, G = ldl(D, U, H)
	Solve LDLTx = b
    
	Returns:
	x[np.array]: result of the linear solve
	'''
    # Forward solve
	n = len(b)
	x = np.zeros(n)
	y = np.zeros(n)
	# Forward solve now correct
	b_copy = np.copy(b)
	for i in range(n):
		y[i] = b_copy[i]
		b_copy[i+1:] -= y[i]*np.dot(U[i+1:, :], G[i, :].T)
	y = y / D_hat
	
	# Backward solve LTx = y
	for i in range(n-1, -1, -1):
		x[i] = y[i] - np.inner(np.dot(U[i+1:, :], G[i, :].T), x[i+1:])
	return x