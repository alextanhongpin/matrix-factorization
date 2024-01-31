# Matrix Factorization with Bias

Similar to 09_matrix_factorization.py, but with user and item bias (TODO: Add reference on why this bias is needed).

## Reference:
- https://github.com/GabrielSandoval/matrix_factorization/blob/master/lib/mf.py
- https://d2l.ai/chapter_recommender-systems/mf.html
- https://medium.com/@maxbrenner-ai/matrix-factorization-for-collaborative-filtering-linear-to-non-linear-models-in-python-5cf54363a03c
- https://github.com/NicolasHug/Surprise/blob/master/surprise/prediction_algorithms/matrix_factorization.pyx
- https://everdark.github.io/k9/notebooks/ml/matrix_factorization/matrix_factorization.nb.html#3_neural_netork_representation

Consider the following matrix of ratings. The rows are the users, while the columns are the items, and the values are the rating 1-5.
In the first row, we see that user rated 5 for the item.

The average rating for this matrix is 2.8 stars (after rounding).


```python
import numpy as np

ratings = np.array(
    [[5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [1, 0, 0, 4], [0, 1, 5, 4]], dtype=float
)
avg_rating = np.mean(ratings[ratings.nonzero()])
np.round(avg_rating, 1)
```




    2.8



We want to learn the bias of the user and the items. In the context of movie recommendation, the user bias can be a user that is picky about a user, and hence gives a lower rating (-0.5) than usual. However, the movie seems to have a of fans giving high rating (1.7).

The final rating after taking into consideration the initial average rating, the user bias and the item bias is:

```
final_rating = 2.8 - 0.5 + 1.7
             = 4.0
```


## Baseline Model

In the baseline model, we cover only the user and item biases.


```python
import matplotlib.pyplot as plt
import numpy as np
```


```python
# Assume we have some ratings matrix R
R = np.array(
    [[5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [1, 0, 0, 4], [0, 1, 5, 4]],
    dtype=np.float32,
)

# Initialize user and item embedding matrices
num_users, num_items = R.shape

# Initialize user and item bias vectors
user_bias = np.zeros(num_users)
item_bias = np.zeros(num_items)

# Initialize average rating.
avg_rating = R[R.nonzero()].mean()

# Define the learning rate and regularization strength
lr = 0.01
reg_strength = 1e-5
losses = []
mask = R > 0
known_ratings = np.sum(mask)
tol = 1e-3

# Define the loss function
def sse_loss(avg_rating, user_bias, item_bias):
    R_hat = avg_rating + user_bias[:, None] + item_bias[None, :]

    loss = np.sum(np.square(R - R_hat)[mask]) / known_ratings + reg_strength * (
        np.sum(np.square(user_bias)) + np.sum(np.square(item_bias))
    )
    return loss


T = 500
# Run the optimization
for t in range(T):
    for u, i in zip(*R.nonzero()):
        b_u = user_bias[u]
        b_i = item_bias[i]
        R_hat = avg_rating + b_u + b_i
        err = R[u, i] - R_hat

        # Update parameters
        user_bias[u] += lr * (err - reg_strength * b_u)
        item_bias[i] += lr * (err - reg_strength * b_i)
    loss = sse_loss(avg_rating, user_bias, item_bias)
    losses.append(loss)
    if loss < tol:
        print(f"Terminating after {t} iterations, loss={loss}")
        break
```


```python
plt.plot(losses)
```




    [<matplotlib.lines.Line2D at 0x12722a560>]




    
![png](10_matrix_factorization_with_bias_files/10_matrix_factorization_with_bias_7_1.png)
    



```python
print("Original ratings:")
print(R)
```

    Original ratings:
    [[5. 3. 0. 1.]
     [4. 0. 0. 1.]
     [1. 1. 0. 5.]
     [1. 0. 0. 4.]
     [0. 1. 5. 4.]]


### Reconstructed Ratings


```python
print("Reconstructed ratings:")
R_hat = avg_rating + user_bias[:, None] + item_bias[None, :]
print(np.round(R_hat, 2))
```

    Reconstructed ratings:
    [[3.39 1.96 5.23 3.6 ]
     [2.39 0.96 4.23 2.59]
     [2.76 1.32 4.59 2.96]
     [2.4  0.97 4.24 2.61]
     [3.14 1.7  4.98 3.34]]



```python
sse_loss(avg_rating, user_bias, item_bias)
```




    2.133126894011934



## SVD

See formula here:

https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/surprise_svd_deep_dive.ipynb


```python
# Assume we have some ratings matrix R
R = np.array(
    [[5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [1, 0, 0, 4], [0, 1, 5, 4]],
    dtype=np.float32,
)

# Initialize user and item embedding matrices
num_users, num_items = R.shape
embedding_dim = 10
U = np.random.normal(size=(num_users, embedding_dim))
V = np.random.normal(size=(num_items, embedding_dim))
non_zero_mask = R > 0
known_ratings = np.sum(non_zero_mask)

# Initialize user and item bias vectors
user_bias = np.zeros(num_users)
item_bias = np.zeros(num_items)

# Initialize average rating.
avg_rating = R[R.nonzero()].mean()

# Define the learning rate and regularization strength
lr = 0.01
reg_strength = 1e-5
tol = 1e-3
losses = []

mask = R > 0
# We take into considerating the known ratings only when calculating errors.
known_ratings = np.sum(mask)

# Define the loss function
def sse_loss(U, V, user_bias, item_bias, avg_rating, R):
    R_hat = U @ V.T + user_bias[:, None] + item_bias[None, :] + avg_rating

    # Squared sum error (SSE) of known ratings.
    loss = np.sum(np.square((R - R_hat)[non_zero_mask])) / known_ratings
    # Add regularization
    loss += reg_strength * (
        np.sum(U ** 2)
        + np.sum(V ** 2)
        + np.sum(user_bias ** 2)
        + np.sum(item_bias ** 2)
    )
    return loss


# Run the optimization
for t in range(500):
    for u, i in zip(*R.nonzero()):
        b_u = user_bias[u]
        b_i = item_bias[i]
        R_hat = U[u] @ V[i].T + b_u + b_i + avg_rating
        err = R[u, i] - R_hat

        # Compute gradients
        grad_U = err * V[i] - reg_strength * U[u]
        grad_V = err * U[u] - reg_strength * V[i]
        grad_user_bias = err - reg_strength * b_u
        grad_item_bias = err - reg_strength * b_i

        # Update parameters
        U[u] += lr * grad_U
        V[i] += lr * grad_V
        user_bias[u] += lr * grad_user_bias
        item_bias[i] += lr * grad_item_bias

    loss = sse_loss(U, V, user_bias, item_bias, avg_rating, R)
    if i % 100 == 0:
        print("Iteration", i, "Loss", loss)
    losses.append(loss)
    if loss < tol:
        print(f"Terminating after {t} iterations, loss={loss}")
        break
```

    Terminating after 52 iterations, loss=0.0009628049799044417



```python
plt.plot(losses)
```




    [<matplotlib.lines.Line2D at 0x1277b3670>]




    
![png](10_matrix_factorization_with_bias_files/10_matrix_factorization_with_bias_14_1.png)
    



```python
print("Original ratings:")
print(R)
```

    Original ratings:
    [[5. 3. 0. 1.]
     [4. 0. 0. 1.]
     [1. 1. 0. 5.]
     [1. 0. 0. 4.]
     [0. 1. 5. 4.]]


### Reconstructed Ratings


```python
print("Reconstructed ratings:")
R_hat = U @ V.T + user_bias[:, None] + item_bias[None, :] + avg_rating
print(np.round(R_hat, 2))
```

    Reconstructed ratings:
    [[5.01 2.99 1.95 0.97]
     [3.99 1.34 4.02 1.04]
     [1.01 0.99 5.47 4.97]
     [1.   2.46 4.04 4.01]
     [5.   1.   5.   4.  ]]



```python
sse_loss(U, V, user_bias, item_bias, avg_rating, R)
```




    0.0009628049799044417



### Output


```python
# If the replace does not work, check if the data type for R matches R_hat, which is float.
# If the data type of the matrix to replace does not match, it will silently failed.
mask = R == 0
R[mask] = R_hat[mask]
np.round(np.clip(R, 0, 5), 1)
```




    array([[5. , 3. , 2. , 1. ],
           [4. , 1.3, 4. , 1. ],
           [1. , 1. , 5. , 5. ],
           [1. , 2.5, 4. , 4. ],
           [5. , 1. , 5. , 4. ]], dtype=float32)




```python
R_hat = (
    U.numpy() @ V.numpy().T
    + avg_rating.numpy()
    + user_bias.numpy()[:, None]
    + item_bias.numpy()[None, :]
)
np.round(R_hat, 2)
```




    array([[ 5.  ,  3.  ,  2.09,  1.  ],
           [ 4.  , -7.36,  6.09,  1.  ],
           [ 1.  ,  1.  ,  4.27,  5.  ],
           [ 1.  ,  1.59,  3.13,  4.  ],
           [ 3.89,  1.  ,  5.  ,  4.  ]], dtype=float32)




```python

```
