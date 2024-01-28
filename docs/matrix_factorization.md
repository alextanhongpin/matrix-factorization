```python
import numpy as np
import pandas as pd
```


```python
movies = [
    "men in black",
    "star trek",
    "ace ventura",
    "braveheart",
    "sense sensibility and snowmen",
    "les miserables",
]
users = ["Sara", "Jesper", "Therese", "Helle", "Pietro", "Ekaterina"]
```


```python
M = pd.DataFrame(
    [
        [5.0, 3.0, 0.0, 2.0, 2.0, 2.0],
        [4.0, 3.0, 4.0, 0.0, 3.0, 3.0],
        [5.0, 2.0, 5.0, 2.0, 1.0, 1.0],
        [3.0, 5.0, 3.0, 0.0, 1.0, 1.0],
        [3.0, 3.0, 3.0, 2.0, 4.0, 5.0],
        [2.0, 3.0, 2.0, 3.0, 5.0, 5.0],
    ],
    columns=movies,
    index=users,
)
```

## Imputation

Solving the problem of zeros in the rating matrix using imputation. 

- calculate the mean of each item (or user) and fill in this mean where there are zeros in each row (or column) of the matrix
- you can normalize each row, such that all elements are centered around zero, so the zeros will become the average


```python
# Calculates the mean of all movies.
r_average = M[M > 0.0].mean()
M[M == 0] = np.NaN
M.fillna(r_average, inplace=True)
```


```python
M["men in black"]["Sara"]
```




    5.0




```python
r_average
```




    men in black                     3.666667
    star trek                        3.166667
    ace ventura                      3.400000
    braveheart                       2.250000
    sense sensibility and snowmen    2.666667
    les miserables                   2.833333
    dtype: float64




```python
M
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>men in black</th>
      <th>star trek</th>
      <th>ace ventura</th>
      <th>braveheart</th>
      <th>sense sensibility and snowmen</th>
      <th>les miserables</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sara</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>3.4</td>
      <td>2.00</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Jesper</th>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.25</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Therese</th>
      <td>5.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>2.00</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Helle</th>
      <td>3.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>2.25</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Pietro</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.00</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Ekaterina</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.00</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
U, Sigma, Vt = np.linalg.svd(M)
```


```python
def rank_k(k):
    U_reduced = np.mat(U[:, :k])  # Select up to k columns.
    Vt_reduced = np.mat(Vt[:k, :])  # Select up to k rows.
    Sigma_reduced = (
        np.eye(k) * Sigma[:k]
    )  # Construct back the diagonal matrix from the given list.
    return U_reduced, Sigma_reduced, Vt_reduced
```


```python
U_reduced, Sigma_reduced, Vt_reduced = rank_k(4)
M_hat = U_reduced * Sigma_reduced * Vt_reduced
np.round(M_hat, 1), M.values
```




    (array([[5. , 3.1, 3.5, 1.8, 1.9, 2.1],
            [4. , 3. , 4. , 2.4, 2.9, 3.1],
            [5. , 2. , 5. , 1.9, 1. , 1. ],
            [3. , 5. , 3. , 2.3, 1. , 0.9],
            [3.1, 2.9, 2.8, 2.4, 4.3, 4.7],
            [1.9, 3.1, 2.2, 2.6, 4.8, 5.2]]),
     array([[5.  , 3.  , 3.4 , 2.  , 2.  , 2.  ],
            [4.  , 3.  , 4.  , 2.25, 3.  , 3.  ],
            [5.  , 2.  , 5.  , 2.  , 1.  , 1.  ],
            [3.  , 5.  , 3.  , 2.25, 1.  , 1.  ],
            [3.  , 3.  , 3.  , 2.  , 4.  , 5.  ],
            [2.  , 3.  , 2.  , 3.  , 5.  , 5.  ]]))




```python
## Predict a rating.
M_hat_matrix = pd.DataFrame(M_hat, columns=movies, index=users).round(2)
M_hat_matrix["ace ventura"]["Sara"]
```




    3.47




```python
# Reduces the size of the decomposed matrices.


def rank_k2(k):
    U_reduced = np.mat(U[:, :k])
    Vt_reduced = np.mat(Vt[:k, :])
    Sigma_reduced = np.eye(k) * Sigma[:k]
    Sigma_sqrt = np.sqrt(Sigma_reduced)
    return U_reduced * Sigma_sqrt, Sigma_sqrt * Vt_reduced
```


```python
U_reduced, Vt_reduced = rank_k2(4)
M_hat = U_reduced * Vt_reduced
M_hat
```




    matrix([[4.96527904, 3.05025112, 3.4699645 , 1.83793044, 1.92302433,
             2.10555193],
            [4.01735702, 2.96224212, 3.9546778 , 2.38970844, 2.89013078,
             3.06685268],
            [4.98346886, 2.02958029, 5.03794119, 1.89657357, 1.0297349 ,
             0.99672747],
            [3.01306011, 4.98240471, 2.97475285, 2.30489446, 1.04428991,
             0.94793143],
            [3.09822111, 2.86285057, 2.80617696, 2.43523214, 4.27650203,
             4.65403727],
            [1.91727352, 3.1213695 , 2.1680414 , 2.60623179, 4.83585754,
             5.23595875]])




```python
Jesper = 1
AceVentura = 2
U_reduced[Jesper] * Vt_reduced[:, AceVentura]
```




    matrix([[3.9546778]])




```python
M_hat[Jesper, AceVentura]
```




    3.9546778000770195




```python
M["ace ventura"]["Jesper"]
```




    4.0




```python
Sara = 0
M_hat[Sara, AceVentura]
```




    3.469964495841926




```python
M["ace ventura"]["Sara"]
```




    3.4



## Adding a new user by folding in

$\hat{i}_\text{new} = r^T_\text{new item} U\Sigma^{-1}$

where
- $i_\text{new}$ is the vector in the reduced space to represent the new item
- $r_\text{new item}$ is the new item user ratings vector
- $\Sigma^{-1}$ is the inverse of the sigma matrix
- $U$ is the user matrix


```python
# We have a new user, Kim that rated the movies too.
r_kim = np.array([4.0, 5.0, 0.0, 3.0, 3.0, 0.0])
```


```python
u_kim = r_kim * Vt_reduced.T * np.linalg.inv(Sigma_reduced)
u_kim
```




    matrix([[-1.41657858,  0.24873359, -1.78276451, -2.17627056]])


