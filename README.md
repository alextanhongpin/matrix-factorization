# matrix-factorization


## Goal

Understand matrix factorization and singular value decomposition (SVD), and how this can be used to create a recommender system. Also, understand why this approach could be better than using the user-based or item-based collaborative filtering. 

## Matrix Factorization 101

1. Factorize a matrix to find out two matrices such that when you multiply them, you will get back the original matrix.
2. Use to discover latent features underlying the interactions between two different kinds of entities.
3. Use to predict ratings (sparse values, or zeros `0` in matrices). Most of the time, the matrix will be sparse because not all users will rate an item. With matrix factorization, we can _predict_ the rating that the user will give on items they have not rated yet.

## Input

Sample input for the algorithm:

## Output

Sample output for the algorithm:

## TODO

1. Run it parallel
2. Hashing the string and store it instead of the whole object (?)

## Keywords

1. SVD (Singular Value Decomposition) - TODO: Definition


## Reference

1. Mining of Massive Datasets. This book is available for free online and contains in-depth chapters on Singular Value Decomposition.

# Matrix factorization

what
- is matrix factorization
- is the input 
- is the output
why
- use matrix factorization
when 
- to use it
- not to use it
- does it work/does it not work
who
how
- to calculate the matrix factorization
- to use it for recommendation

Factorizing a matrix
- We have a rating matrix R with n rows and m columns. 
- The columns is for user
- The rows is for items
- So we have a n x m matrix (read n by m).
- We can decompose R into U x V
- U will be n x d matrix
- V will be d x m matrix
- U is the user-feature matrix
- V is the item-feature matrix

Singular value decomposition (SVD) is an algorithm commonly used for matrix factorization. 
We can use it to find items to recommend to users.

M - A matrix you want to decompose; in your case, it’s the rating matrix
U - user feature matrix
Sigma - weights diagonal
Vt - item feature matrix


## Questions

how to scale collaborative filtering/matrix factorisation in production for large number of users?

how to update them real-time or close to real-time?

## References

- https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
- http://www.albertauyeung.com/post/python-matrix-factorization/
- https://beckernick.github.io/matrix-factorization-recommender/
- https://lazyprogrammer.me/tutorial-on-collaborative-filtering-and-matrix-factorization-in-python/
