```python
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
```


```python
# Seed to get consistent values
np.random.seed(seed=1)

# Generate sample data
data = np.random.randint(5, size=(5, 10))
data = np.array([[5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [1, 0, 0, 4], [0, 1, 5, 4]])

print("original: {}".format(data))

# Get the mean
mean = np.mean(data, axis=1)

# Demean the data (so that the mean is always zero)
data_demeaned = data - mean.reshape(-1, 1)

# k is the latent features. The value k must be between
# 1 and min(data_demeaned.shape) - 1
k = min(data_demeaned.shape) - 1
# Get the Singular Value Decomposition
U, sigma, Vt = svds(data_demeaned, k=k)

sigma = np.diag(sigma)

print("U: \n{}\n".format(U))
print("sigma: \n{}\n".format(sigma))
print("Vt: \n{}\n".format(Vt))

predicted_ratings = np.dot(np.dot(U, sigma), Vt) + mean.reshape(-1, 1)
print("predicted_ratings: \n{}\n".format(predicted_ratings))
```

    original: [[5 3 0 1]
     [4 0 0 1]
     [1 1 0 5]
     [1 0 0 4]
     [0 1 5 4]]
    U: 
    [[-0.10634084  0.18540429 -0.59393667]
     [ 0.80626825  0.30892111 -0.3747321 ]
     [-0.31668765  0.69499063  0.25779275]
     [ 0.0876219   0.60844446  0.21147617]
     [ 0.480265   -0.13029336  0.62899588]]
    
    sigma: 
    [[ 2.1196438   0.          0.        ]
     [ 0.          4.91892905  0.        ]
     [ 0.          0.          6.2698682 ]]
    
    Vt: 
    [[ 0.44335192 -0.79258604  0.4136403  -0.06440618]
     [ 0.20568993 -0.27108679 -0.63140444  0.6968013 ]
     [-0.71493407 -0.21981658  0.42453521  0.51021544]]
    
    predicted_ratings: 
    [[  5.00000000e+00   3.00000000e+00  -4.44089210e-16   1.00000000e+00]
     [  4.00000000e+00   4.44089210e-16  -2.22044605e-16   1.00000000e+00]
     [  1.00000000e+00   1.00000000e+00  -2.22044605e-16   5.00000000e+00]
     [  1.00000000e+00  -4.44089210e-16  -4.44089210e-16   4.00000000e+00]
     [  1.77635684e-15   1.00000000e+00   5.00000000e+00   4.00000000e+00]]
    

