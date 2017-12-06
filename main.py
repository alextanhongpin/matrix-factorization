import time
import numpy as np
from numba import jit

@jit(nopython=True, parallel=True)
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P, Q)
        e = 0

        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)

                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q.T


R = [[5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4]]

R = np.array(R)

N = len(R)
M = len(R[0])
K = 2

# P = np.random.rand(N, K)
# Q = np.random.rand(M, K)

P = np.array([[ 0.76429108,  0.68098794],
 [ 0.33121903,  0.56332544],
 [ 0.00385484,  0.52690095],
 [ 0.40541877,  0.49737146],
 [ 0.17445866,  0.52815968]])
Q = np.array([[ 0.00909807,  0.57423495],
 [ 0.57250499,  0.06490164],
 [ 0.39694569,  0.77008759],
 [ 0.95925171,  0.1277469 ]])

start = time.time()
nP, nQ = matrix_factorization(R, P, Q, K)
nR = np.dot(nP, nQ.T)
elapsed = time.time() - start
print('Elapsed time: {}'.format(elapsed))
print(nR)