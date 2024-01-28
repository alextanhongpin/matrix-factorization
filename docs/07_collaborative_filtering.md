# Collaborative Filtering

In collaborative filtering, we observe similar users or items when making predictions on the **ratings**.



```python
import numpy as np
import pandas as pd
```

    /var/folders/7m/74_ct3hx33d878n626w1wxyc0000gn/T/ipykernel_69303/1662815981.py:2: DeprecationWarning: 
    Pyarrow will become a required dependency of pandas in the next major release of pandas (pandas 3.0),
    (to allow more performant data types, such as the Arrow string type, and better interoperability with other libraries)
    but was not found to be installed on your system.
    If this would cause problems for you,
    please provide us feedback at https://github.com/pandas-dev/pandas/issues/54466
            
      import pandas as pd


## Preparing Dataset

Below we have a rating matrix. The rows are users, and the columns are movie genre.  We want to predict the rating of `anime` genre for `user b`. The unknown ratings are filled with `np.nan`.


```python
critics = {
    "Lisa Rose": {
        "Lady in the Water": 2.5,
        "Snakes on a Plane": 3.5,
        "Just My Luck": 3.0,
        "Superman Returns": 3.5,
        "You, Me and Dupree": 2.5,
        "The Night Listener": 3.0,
    },
    "Gene Seymour": {
        "Lady in the Water": 3.0,
        "Snakes on a Plane": 3.5,
        "Just My Luck": 1.5,
        "Superman Returns": 5.0,
        "The Night Listener": 3.0,
        "You, Me and Dupree": 3.5,
    },
    "Michael Phillips": {
        "Lady in the Water": 2.5,
        "Snakes on a Plane": 3.0,
        "Superman Returns": 3.5,
        "The Night Listener": 4.0,
    },
    "Claudia Puig": {
        "Snakes on a Plane": 3.5,
        "Just My Luck": 3.0,
        "The Night Listener": 4.5,
        "Superman Returns": 4.0,
        "You, Me and Dupree": 2.5,
    },
    "Mick LaSalle": {
        "Lady in the Water": 3.0,
        "Snakes on a Plane": 4.0,
        "Just My Luck": 2.0,
        "Superman Returns": 3.0,
        "The Night Listener": 3.0,
        "You, Me and Dupree": 2.0,
    },
    "Jack Matthews": {
        "Lady in the Water": 3.0,
        "Snakes on a Plane": 4.0,
        "The Night Listener": 3.0,
        "Superman Returns": 5.0,
        "You, Me and Dupree": 3.5,
    },
    "Toby": {
        "Snakes on a Plane": 4.5,
        "You, Me and Dupree": 1.0,
        "Superman Returns": 4.0,
    },
}

df = pd.DataFrame(critics).T
df
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
      <th>Lady in the Water</th>
      <th>Snakes on a Plane</th>
      <th>Just My Luck</th>
      <th>Superman Returns</th>
      <th>You, Me and Dupree</th>
      <th>The Night Listener</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lisa Rose</th>
      <td>2.5</td>
      <td>3.5</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>2.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Gene Seymour</th>
      <td>3.0</td>
      <td>3.5</td>
      <td>1.5</td>
      <td>5.0</td>
      <td>3.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Michael Phillips</th>
      <td>2.5</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>3.5</td>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Claudia Puig</th>
      <td>NaN</td>
      <td>3.5</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.5</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>Mick LaSalle</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Jack Matthews</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>3.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Toby</th>
      <td>NaN</td>
      <td>4.5</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



We convert all `NaN` to 0.


```python
df.fillna(0, inplace=True)
```


```python
df
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
      <th>Lady in the Water</th>
      <th>Snakes on a Plane</th>
      <th>Just My Luck</th>
      <th>Superman Returns</th>
      <th>You, Me and Dupree</th>
      <th>The Night Listener</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lisa Rose</th>
      <td>2.5</td>
      <td>3.5</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>2.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Gene Seymour</th>
      <td>3.0</td>
      <td>3.5</td>
      <td>1.5</td>
      <td>5.0</td>
      <td>3.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Michael Phillips</th>
      <td>2.5</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Claudia Puig</th>
      <td>0.0</td>
      <td>3.5</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.5</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>Mick LaSalle</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Jack Matthews</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>3.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Toby</th>
      <td>0.0</td>
      <td>4.5</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



We won't use the default `.corr()` method from `pandas`, because it does not take zeros into account.
We want to skip the row/col with zeros when calculating the Pearson correlation.


```python
df.T.corr()
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
      <th>Lisa Rose</th>
      <th>Gene Seymour</th>
      <th>Michael Phillips</th>
      <th>Claudia Puig</th>
      <th>Mick LaSalle</th>
      <th>Jack Matthews</th>
      <th>Toby</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lisa Rose</th>
      <td>1.000000</td>
      <td>0.396059</td>
      <td>0.510754</td>
      <td>0.701287</td>
      <td>0.594089</td>
      <td>0.331618</td>
      <td>0.795744</td>
    </tr>
    <tr>
      <th>Gene Seymour</th>
      <td>0.396059</td>
      <td>1.000000</td>
      <td>0.531008</td>
      <td>0.236088</td>
      <td>0.411765</td>
      <td>0.958785</td>
      <td>0.703861</td>
    </tr>
    <tr>
      <th>Michael Phillips</th>
      <td>0.510754</td>
      <td>0.531008</td>
      <td>1.000000</td>
      <td>0.328336</td>
      <td>0.783869</td>
      <td>0.604105</td>
      <td>0.374818</td>
    </tr>
    <tr>
      <th>Claudia Puig</th>
      <td>0.701287</td>
      <td>0.236088</td>
      <td>0.328336</td>
      <td>1.000000</td>
      <td>0.152763</td>
      <td>0.170544</td>
      <td>0.389391</td>
    </tr>
    <tr>
      <th>Mick LaSalle</th>
      <td>0.594089</td>
      <td>0.411765</td>
      <td>0.783869</td>
      <td>0.152763</td>
      <td>1.000000</td>
      <td>0.564764</td>
      <td>0.640828</td>
    </tr>
    <tr>
      <th>Jack Matthews</th>
      <td>0.331618</td>
      <td>0.958785</td>
      <td>0.604105</td>
      <td>0.170544</td>
      <td>0.564764</td>
      <td>1.000000</td>
      <td>0.687269</td>
    </tr>
    <tr>
      <th>Toby</th>
      <td>0.795744</td>
      <td>0.703861</td>
      <td>0.374818</td>
      <td>0.389391</td>
      <td>0.640828</td>
      <td>0.687269</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
def custom_pearson_correlation(m, n):
    # Skip zeros (unrated).
    mask = list(set(np.where(m != 0)[0]) & set(np.where(n != 0)[0]))

    m = m[mask]
    n = n[mask]

    return pd.Series(m).corr(pd.Series(n))


df.T.corr(custom_pearson_correlation)
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
      <th>Lisa Rose</th>
      <th>Gene Seymour</th>
      <th>Michael Phillips</th>
      <th>Claudia Puig</th>
      <th>Mick LaSalle</th>
      <th>Jack Matthews</th>
      <th>Toby</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lisa Rose</th>
      <td>1.000000</td>
      <td>0.396059</td>
      <td>0.404520</td>
      <td>0.566947</td>
      <td>0.594089</td>
      <td>0.747018</td>
      <td>0.991241</td>
    </tr>
    <tr>
      <th>Gene Seymour</th>
      <td>0.396059</td>
      <td>1.000000</td>
      <td>0.204598</td>
      <td>0.314970</td>
      <td>0.411765</td>
      <td>0.963796</td>
      <td>0.381246</td>
    </tr>
    <tr>
      <th>Michael Phillips</th>
      <td>0.404520</td>
      <td>0.204598</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-0.258199</td>
      <td>0.134840</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>Claudia Puig</th>
      <td>0.566947</td>
      <td>0.314970</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.566947</td>
      <td>0.028571</td>
      <td>0.893405</td>
    </tr>
    <tr>
      <th>Mick LaSalle</th>
      <td>0.594089</td>
      <td>0.411765</td>
      <td>-0.258199</td>
      <td>0.566947</td>
      <td>1.000000</td>
      <td>0.211289</td>
      <td>0.924473</td>
    </tr>
    <tr>
      <th>Jack Matthews</th>
      <td>0.747018</td>
      <td>0.963796</td>
      <td>0.134840</td>
      <td>0.028571</td>
      <td>0.211289</td>
      <td>1.000000</td>
      <td>0.662849</td>
    </tr>
    <tr>
      <th>Toby</th>
      <td>0.991241</td>
      <td>0.381246</td>
      <td>-1.000000</td>
      <td>0.893405</td>
      <td>0.924473</td>
      <td>0.662849</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
def similar_to(df, user, n=5):
    """
    Finding the top-n users is as simple as just computing the pearson correlation scores,
    and returning the sorted result.
    """
    return sorted(
        df.T.corr(custom_pearson_correlation)[user].drop(user).items(),
        key=lambda t: t[1],
        reverse=True,
    )[:n]
```


```python
similar_to(df, "Toby")
```




    [('Lisa Rose', 0.9912407071619304),
     ('Mick LaSalle', 0.924473451641905),
     ('Claudia Puig', 0.8934051474415642),
     ('Jack Matthews', 0.6628489803598703),
     ('Gene Seymour', 0.3812464258315117)]




```python
# For item based collaborative filtering, we just transpose the df.
similar_to(df.T, "Just My Luck")
```




    [('The Night Listener', 0.5555555555555556),
     ('Snakes on a Plane', -0.3333333333333333),
     ('Superman Returns', -0.42289003161103106),
     ('You, Me and Dupree', -0.4856618642571827),
     ('Lady in the Water', -0.944911182523068)]




```python
def recommend(df, user):
    similarity_scores = similar_to(df, user)
    recs = []

    # Only select movies that has np.nan ratings.
    not_watched = df.columns[df.loc[user] == 0]

    for movie in not_watched:
        # Ratings for the movie from other users.
        rated_by_user = dict(df[movie].fillna(0))

        sum_weight = 0
        sum_rating = 0

        for user, weight in similarity_scores:
            # Ignore users that did not give rating.
            rating = rated_by_user[user]
            if rating == 0:
                continue

            sum_weight += weight
            sum_rating += weight * rating

        recs.append((movie, sum_rating / sum_weight))

    # Sort by rating, in descending order (highest to lowest rating)
    return sorted(recs, key=lambda t: t[1], reverse=True)
```


```python
recommend(df, "Toby")
```




    [('The Night Listener', 3.3477895267131013),
     ('Lady in the Water', 2.8325499182641622),
     ('Just My Luck', 2.5309807037655645)]




```python
recommend(df, "Michael Phillips")
```




    [('Just My Luck', 2.963951538816175),
     ('You, Me and Dupree', 2.8153523713809516)]



## Cold Start

A cold start problem is when we do not have enough information from a new user to provide recommendation. This cannot be solve by machine learning. What we can do is just taking existing information about the product to make recommendation, e.g. how popular or trending a product is.


Below, we will just show how we recommend by converting the ratings to `like` and `dislike`, and calculating the probability that a user will like the movie.

We convert the ratings into like/dislike. Anything below 2.5 will be treated as dislike.


```python
likes = np.where(df <= 2.5, 0, 1)
like_df = pd.DataFrame(likes, index=df.index, columns=df.columns)
like_df
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
      <th>Lady in the Water</th>
      <th>Snakes on a Plane</th>
      <th>Just My Luck</th>
      <th>Superman Returns</th>
      <th>You, Me and Dupree</th>
      <th>The Night Listener</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lisa Rose</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Gene Seymour</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Michael Phillips</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Claudia Puig</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Mick LaSalle</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Jack Matthews</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Toby</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
like_df.count(axis=0)
```




    Lady in the Water     7
    Snakes on a Plane     7
    Just My Luck          7
    Superman Returns      7
    You, Me and Dupree    7
    The Night Listener    7
    dtype: int64




```python
like_df.sum(axis=0)
```




    Lady in the Water     3
    Snakes on a Plane     7
    Just My Luck          2
    Superman Returns      7
    You, Me and Dupree    2
    The Night Listener    6
    dtype: int64



We can see that `Superman Returns` is liked by all users that provided ratings. However, we cannot say that the probability of recommending it is 100%.

We can calculate the probability that I will like/dislike it by just adding 2 new implicit feedback, 1 like and 1 dislike and see how the ratings changes:

```
# For Superman Returns
prob = (7 + 1) / (7 + 2)
     = 0.88
```


```python
sorted(
    ((like_df.sum(axis=0) + 1) / (like_df.count(axis=0) + 2)).items(),
    key=lambda t: t[1],
    reverse=True,
)
```




    [('Snakes on a Plane', 0.8888888888888888),
     ('Superman Returns', 0.8888888888888888),
     ('The Night Listener', 0.7777777777777778),
     ('Lady in the Water', 0.4444444444444444),
     ('Just My Luck', 0.3333333333333333),
     ('You, Me and Dupree', 0.3333333333333333)]


