## Matrix Factorization

https://blog.fastforwardlabs.com/2018/04/10/pytorch-for-recommenders-101.html


```python
import numpy as np

ratings = np.array(
    [[5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [1, 0, 0, 4], [0, 1, 5, 4]], dtype=float
)
ratings
```




    array([[5., 3., 0., 1.],
           [4., 0., 0., 1.],
           [1., 1., 0., 5.],
           [1., 0., 0., 4.],
           [0., 1., 5., 4.]])




```python
import torch
from torch.autograd import Variable


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=3):
        super().__init__()

        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=True)

    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)

    def predict(self, user, item):
        return self.forward(user, item)
```


```python
n_users = len(ratings)
n_items = len(ratings[0])
model = MatrixFactorization(n_users, n_items, n_factors=3)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for i in range(500):
    done = False
    for user, item in zip(*ratings.nonzero()):
        # get user, item and rating data
        rating = Variable(torch.FloatTensor([ratings[user, item]]))
        user = Variable(torch.LongTensor([int(user)]))
        item = Variable(torch.LongTensor([int(item)]))

        # predict
        prediction = model(user, item)
        loss = loss_fn(prediction, rating)
        optimizer.zero_grad()

        # backpropagate
        loss.backward()

        # update weights
        optimizer.step()
```


```python
predicted_ratings = np.zeros((n_users, n_items))
for u in range(n_users):
    for i in range(n_items):
        predicted_ratings[u, i] = model.predict(
            torch.LongTensor([u]), torch.LongTensor([i])
        )
np.round(predicted_ratings, 2)
```




    array([[ 5.03,  2.9 ,  3.15,  1.01],
           [ 4.  , -2.1 , -1.68,  1.02],
           [ 0.97,  1.12,  5.62,  4.94],
           [ 1.01,  0.21,  3.79,  3.97],
           [-0.94,  0.95,  4.93,  4.11]])




```python
np.round(ratings, 2)
```




    array([[5., 3., 0., 1.],
           [4., 0., 0., 1.],
           [1., 1., 0., 5.],
           [1., 0., 0., 4.],
           [0., 1., 5., 4.]])



## Dense Feedforward Neural Network


```python
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class DenseNet(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors, H1, D_out):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.linear1 = torch.nn.Linear(n_factors * 2, H1)
        self.linear2 = torch.nn.Linear(H1, D_out)

    def forward(self, users, items):
        users_embedding = self.user_factors(users)
        items_embedding = self.item_factors(items)
        # Concatenate users and items embeddings to form input.
        X = torch.cat([users_embedding, items_embedding], 1)
        h1_relu = F.relu(self.linear1(X))
        output_scores = self.linear2(h1_relu)
        return output_scores

    def predict(self, users, items):
        output_scores = self.forward(users, items)
        return output_scores
```


```python
n_users = len(ratings)
n_items = len(ratings[0])
n_factors = 3
H1 = 16
D_out = 1
model = DenseNet(n_users, n_items, n_factors, H1, D_out)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for i in range(500):
    done = False
    for user, item in zip(*ratings.nonzero()):
        # get user, item and rating data
        rating = Variable(torch.FloatTensor([ratings[user, item]]))
        user = Variable(torch.LongTensor([int(user)]))
        item = Variable(torch.LongTensor([int(item)]))

        # predict
        prediction = model(user, item)
        loss = loss_fn(prediction, rating)
        optimizer.zero_grad()

        # backpropagate
        loss.backward()

        # update weights
        optimizer.step()
```


```python
predicted_ratings = np.zeros((n_users, n_items))
for u in range(n_users):
    for i in range(n_items):
        predicted_ratings[u, i] = model.predict(
            torch.LongTensor([u]), torch.LongTensor([i])
        )
np.round(predicted_ratings, 2)
```




    array([[4.97, 3.01, 5.25, 1.05],
           [4.01, 2.54, 4.42, 0.99],
           [1.01, 1.  , 1.13, 5.02],
           [1.  , 0.94, 1.49, 3.96],
           [3.89, 0.99, 5.01, 4.01]])




```python
np.round(ratings, 2)
```




    array([[5., 3., 0., 1.],
           [4., 0., 0., 1.],
           [1., 1., 0., 5.],
           [1., 0., 0., 4.],
           [0., 1., 5., 4.]])




```python
loss_fn(
    torch.Tensor(predicted_ratings[ratings.nonzero()]),
    torch.Tensor(ratings[ratings.nonzero()]),
)
```




    tensor(0.0005)


