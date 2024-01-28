```python
import numpy as np
import pandas as pd
```


```python
data = pd.read_csv('data/ratings.csv')[:100]
data.head()
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1129</td>
      <td>2.0</td>
      <td>1260759185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1172</td>
      <td>4.0</td>
      <td>1260759205</td>
    </tr>
  </tbody>
</table>
</div>




```python
movie_data = pd.read_csv('data/movies.csv')
movie_data.head()
```

          movieId                                              title  \
    0           1                                   Toy Story (1995)   
    1           2                                     Jumanji (1995)   
    2           3                            Grumpier Old Men (1995)   
    3           4                           Waiting to Exhale (1995)   
    4           5                 Father of the Bride Part II (1995)   
    5           6                                        Heat (1995)   
    6           7                                     Sabrina (1995)   
    7           8                                Tom and Huck (1995)   
    8           9                                Sudden Death (1995)   
    9          10                                   GoldenEye (1995)   
    10         11                     American President, The (1995)   
    11         12                 Dracula: Dead and Loving It (1995)   
    12         13                                       Balto (1995)   
    13         14                                       Nixon (1995)   
    14         15                            Cutthroat Island (1995)   
    15         16                                      Casino (1995)   
    16         17                       Sense and Sensibility (1995)   
    17         18                                  Four Rooms (1995)   
    18         19              Ace Ventura: When Nature Calls (1995)   
    19         20                                 Money Train (1995)   
    20         21                                  Get Shorty (1995)   
    21         22                                     Copycat (1995)   
    22         23                                   Assassins (1995)   
    23         24                                      Powder (1995)   
    24         25                           Leaving Las Vegas (1995)   
    25         26                                     Othello (1995)   
    26         27                                Now and Then (1995)   
    27         28                                  Persuasion (1995)   
    28         29  City of Lost Children, The (Cité des enfants p...   
    29         30  Shanghai Triad (Yao a yao yao dao waipo qiao) ...   
    ...       ...                                                ...   
    9095   159690  Teenage Mutant Ninja Turtles: Out of the Shado...   
    9096   159755          Popstar: Never Stop Never Stopping (2016)   
    9097   159858                             The Conjuring 2 (2016)   
    9098   159972                     Approaching the Unknown (2016)   
    9099   160080                                Ghostbusters (2016)   
    9100   160271                        Central Intelligence (2016)   
    9101   160438                                Jason Bourne (2016)   
    9102   160440                             The Maid's Room (2014)   
    9103   160563                        The Legend of Tarzan (2016)   
    9104   160565                    The Purge: Election Year (2016)   
    9105   160567              Mike & Dave Need Wedding Dates (2016)   
    9106   160590                         Survive and Advance (2013)   
    9107   160656                                    Tallulah (2016)   
    9108   160718                                       Piper (2016)   
    9109   160954                                       Nerve (2016)   
    9110   161084                       My Friend Rockefeller (2015)   
    9111   161155                                   Sunspring (2016)   
    9112   161336                  Author: The JT LeRoy Story (2016)   
    9113   161582                          Hell or High Water (2016)   
    9114   161594               Kingsglaive: Final Fantasy XV (2016)   
    9115   161830                                        Body (2015)   
    9116   161918                Sharknado 4: The 4th Awakens (2016)   
    9117   161944              The Last Brickmaker in America (2001)   
    9118   162376                                    Stranger Things   
    9119   162542                                      Rustom (2016)   
    9120   162672                                Mohenjo Daro (2016)   
    9121   163056                               Shin Godzilla (2016)   
    9122   163949  The Beatles: Eight Days a Week - The Touring Y...   
    9123   164977                           The Gay Desperado (1936)   
    9124   164979                              Women of '69, Unboxed   
    
                                                   genres  
    0         Adventure|Animation|Children|Comedy|Fantasy  
    1                          Adventure|Children|Fantasy  
    2                                      Comedy|Romance  
    3                                Comedy|Drama|Romance  
    4                                              Comedy  
    5                               Action|Crime|Thriller  
    6                                      Comedy|Romance  
    7                                  Adventure|Children  
    8                                              Action  
    9                           Action|Adventure|Thriller  
    10                               Comedy|Drama|Romance  
    11                                      Comedy|Horror  
    12                       Adventure|Animation|Children  
    13                                              Drama  
    14                           Action|Adventure|Romance  
    15                                        Crime|Drama  
    16                                      Drama|Romance  
    17                                             Comedy  
    18                                             Comedy  
    19                 Action|Comedy|Crime|Drama|Thriller  
    20                              Comedy|Crime|Thriller  
    21                Crime|Drama|Horror|Mystery|Thriller  
    22                              Action|Crime|Thriller  
    23                                       Drama|Sci-Fi  
    24                                      Drama|Romance  
    25                                              Drama  
    26                                     Children|Drama  
    27                                      Drama|Romance  
    28             Adventure|Drama|Fantasy|Mystery|Sci-Fi  
    29                                        Crime|Drama  
    ...                                               ...  
    9095                          Action|Adventure|Comedy  
    9096                                           Comedy  
    9097                                           Horror  
    9098                            Drama|Sci-Fi|Thriller  
    9099                      Action|Comedy|Horror|Sci-Fi  
    9100                                    Action|Comedy  
    9101                                           Action  
    9102                                         Thriller  
    9103                                 Action|Adventure  
    9104                             Action|Horror|Sci-Fi  
    9105                                           Comedy  
    9106                               (no genres listed)  
    9107                                            Drama  
    9108                                        Animation  
    9109                                   Drama|Thriller  
    9110                                      Documentary  
    9111                                           Sci-Fi  
    9112                                      Documentary  
    9113                                      Crime|Drama  
    9114  Action|Adventure|Animation|Drama|Fantasy|Sci-Fi  
    9115                            Drama|Horror|Thriller  
    9116                   Action|Adventure|Horror|Sci-Fi  
    9117                                            Drama  
    9118                                            Drama  
    9119                                 Romance|Thriller  
    9120                          Adventure|Drama|Romance  
    9121                  Action|Adventure|Fantasy|Sci-Fi  
    9122                                      Documentary  
    9123                                           Comedy  
    9124                                      Documentary  
    
    [9125 rows x 3 columns]



```python
# Create rating matrix of shape (m x u) with rows as movies and columns as users
ratings_mat = np.ndarray(
    shape=(np.max(data.movieId.values), np.max(data.userId.values)),
    dtype=np.uint8)

ratings_mat[data.movieId.values - 1, data.userId.values - 1] = data.rating.values
```


```python
# Normalize matrix (subtract mean off)
normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T
print(normalised_mat)
```

    [[ 0.          0.          0.        ]
     [ 0.          0.          0.        ]
     [-3.         -3.          6.        ]
     ..., 
     [-1.33333333 -1.33333333  2.66666667]
     [ 1.33333333 -0.66666667 -0.66666667]
     [ 2.         -1.         -1.        ]]



```python
# Compute svd
num_movies = data.shape[0] - 1
A = normalised_mat.T / np.sqrt(num_movies)
U, S, V = np.linalg.svd(A)
print(U, S, V)
```

    [[ 0.35654013  0.73453782  0.57735027]
     [-0.81439847 -0.0584961   0.57735027]
     [ 0.45785835 -0.67604172  0.57735027]] [  3.02876736e+02   2.74557212e+02   9.39216305e-13] [[ -2.98538890e-15   3.43800415e-16   1.36738286e-03 ...,   6.07725715e-04
        2.36621878e-04   3.54932818e-04]
     [ -1.06137294e-15  -5.59991992e-16  -2.22723248e-03 ...,  -9.89881101e-04
        5.37766447e-04   8.06649670e-04]
     [  9.89264167e-01   1.39802496e-01  -6.90506663e-06 ...,   9.05634998e-04
        1.39452279e-05   4.17808430e-06]
     ..., 
     [ -1.02167246e-03   7.72501302e-04  -1.62574570e-03 ...,   9.99997186e-01
        3.64475714e-07   5.73877716e-07]
     [ -2.02054491e-05   3.58568058e-05   5.71815199e-04 ...,   9.03861692e-07
        9.99999663e-01  -5.14639858e-07]
     [ -1.37527632e-05   5.61594187e-05   8.74435377e-04 ...,   1.37087751e-06
       -5.04834639e-07   9.99999228e-01]]



```python
def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(movie_data[movie_data.movieId == movie_id].title.values[0]))
    for id in top_indexes + 1:
        if not movie_data[movie_data.movieId == id].empty:
            print(movie_data[movie_data.movieId == id].title.values[0])
```


```python
k = 50
movie_id = 2
top_n = 10

sliced = V.T[:, :k]
indexes = top_cosine_similarity(sliced, movie_id, top_n)
print_similar_movies(movie_data, movie_id, indexes)
```

    Recommendations for Jumanji (1995): 
    
    Jumanji (1995)
    Toy Story (1995)
    Room at the Top (1959)
    Virus (1999)
    White Men Can't Jump (1992)
    Dumb & Dumber (Dumb and Dumber) (1994)
    French Kiss (1995)
    I Saw What You Did (1965)


    /usr/local/lib/python3.6/site-packages/ipykernel_launcher.py:5: RuntimeWarning: invalid value encountered in true_divide
      """

