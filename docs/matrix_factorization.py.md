# Matrix Factorization

The problem we propose to address here is that of __rating prediction__. The data we have is a rating history: ratings of users for items in the interval [1,5] (1 to 5 stars rating). We can put all this data into a sparse matrix called $R$.

The matrix $R$ is sparse (more than 99% of the entries are missing), and our goal is to __predict__ the missing entries. One of the solution for this rating prediction is _matrix factorization_. This is fundamentally linked to SVD, which stands for Singular Value Decomposition.

Here is the matrix factorization:

$R = M\sum U^T$

To be clear, SVD is an algorithm that takes the matrix $R$ as an input, and it gives you $M$, $\sum$ and $U$.

## References:

1. http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
2. http://www.albertauyeung.com/post/python-matrix-factorization/
3. https://arxiv.org/pdf/1503.07475.pdf


```python
import numpy as np
```


```python
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
                                                                                  
```


```python
R = [[5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4]]

R = np.array(R)

N = len(R)
M = len(R[0])
K = 2

np.random.seed(1)

P = np.random.rand(N, K)
Q = np.random.rand(M, K)

nP, nQ = matrix_factorization(R, P, Q, K)
nR = np.dot(nP, nQ.T)
print(nR)
```

    [[ 5.03195688  2.83201472  5.4477017   0.99689041]
     [ 3.93779077  2.21915034  4.3989906   0.99779349]
     [ 1.11553482  0.69186028  4.16784453  4.96378344]
     [ 0.94429841  0.58260453  3.38688405  3.97561872]
     [ 2.43705479  1.41955041  4.85567822  4.03535322]]



```python
class MatrixFactorization:
    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty entries in a matrix
        
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimension
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        
        self.R = R 
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        
    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])
        
        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        # Perform stochastic gradient descent for number of iterations
        
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i + 1) % 10 == 0:
                print('Iterations: {}; error = {}'.format(i + 1, mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic gradient descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            
            # Update bias
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_u[j])
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])
        
    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j,:].T)
        return prediction
    
    def full_matrix(self):
        """
        Compute the full matrix using the resultant biases, P and Q
        """
        return mf.b + mf.b_u[:, np.newaxis] + mf.b_i[np.newaxis:,] + mf.P.dot(mf.Q.T)
```


```python
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

mf = MatrixFactorization(R, K=5, alpha=0.1, beta=0.01, iterations=10)
res = mf.train()
print(mf.full_matrix())
```

    Iterations: 10; error = 0.09607282078386882
    [[ 4.98632057  3.00079431  3.35652779  1.00834942]
     [ 3.99098164  2.07780363  3.12441336  1.01347016]
     [ 0.95429424  1.03175376  3.47626225  4.97060753]
     [ 1.05347149  1.17222442  2.91033953  4.01960965]
     [ 2.74647337  1.01132162  4.96684246  4.01664825]]

