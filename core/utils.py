import numpy as np
def house(u):
    dum = np.dot(u[1:], u[1:])
    c = np.sqrt(u[0]*u[0]+dum)
    h = np.copy(u)
    h[0] = np.sign(u[0])*c + u[0]
    h = h / np.sqrt(h[0]*h[0]+dum)
    return (-1*np.sign(u[0])*c, h)

def planerot(x):
    # Assumes that x has length 2
    assert len(x) == 2
    if x[1] == 0:
        return (np.eye(2), x)
    r = np.linalg.norm(x)
    G = np.array([[x[0],x[1]], [-1*x[1], x[0]]])/r
    return (G, np.array([r, 0]))

def inertia_qr(ratio, D_hat, tol=1e-7):
    D_hat = D_hat / ratio
    neg, zero, pos = 0, 0, 0
    for d in D_hat:
        if abs(d) < tol:
            zero += 1
        elif d > 0:
            pos += 1
        else:
            neg += 1
    return (pos, zero, neg)

def inertia_ldl(D_hat, tol=1e-7):
    neg, zero, pos = 0, 0, 0
    for d in D_hat:
        if abs(d) < tol:
            zero += 1
        elif d > 0:
            pos += 1
        else:
            neg += 1
    return (pos, zero, neg)

def comp_error(D, U, H, mu, x):
    y = np.dot(U, np.dot(H, np.dot(U.T, x)))
    y += np.multiply(D, x)
    return np.linalg.norm(y - mu*x)