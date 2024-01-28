# Matrix Factorization Example using VowpalWabbit

Reference:
https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Matrix-factorization-example


```python
!./mf-setup.sh
```

    File ml-100k.zip exists.
    Archive:  ml-100k.zip



```python
!./mf-run.sh
```

    creating quadratic features for pairs: ui
    using l2 regularization = 0.001
    final_regressor = movielens.reg
    using cache_file = movielens.cache
    ignoring text input in favor of cache input
    num sources = 1
    Num weight bits = 18
    learning rate = 0.015
    initial_t = 1
    power_t = 0
    decay_learning_rate = 0.97
    Enabled learners: rank, scorer-identity, count_label
    Input label = SIMPLE
    Output pred = SCALAR
    average  since         example        example        current        current  current
    loss     last          counter         weight          label        predict features
    23.47829 23.47829            1            1.0         5.0000         0.1546       23
    15.52144 7.564596            2            2.0         3.0000         0.2496       23
    12.98887 10.45630            4            4.0         3.0000         0.3375       23
    11.25910 9.529338            8            8.0         1.0000         0.6490       23
    12.26689 13.27468           16           16.0         3.0000         1.1538       23
    8.234041 4.201188           32           32.0         2.0000         1.6854       23
    5.874644 3.515247           64           64.0         1.0000         2.7666       23
    3.985317 2.095991          128          128.0         4.0000         3.1239       23
    2.961340 1.937362          256          256.0         4.0000         2.3446       23
    2.442762 1.924185          512          512.0         5.0000         2.9704       23
    1.816145 1.189528         1024         1024.0         3.0000         3.6345       23
    1.634253 1.452361         2048         2048.0         3.0000         3.9241       23
    1.408795 1.183336         4096         4096.0         4.0000         3.6946       23
    1.225044 1.041293         8192         8192.0         5.0000         3.7049       23
    1.167822 1.110601        16384        16384.0         3.0000         3.2662       23
    1.089689 1.011556        32768        32768.0         2.0000         3.8134       23
    1.034541 0.979392        65536        65536.0         4.0000         4.1479       23
    0.984682 0.984682       131072       131072.0         3.0000         3.5838       23 h
    0.940430 0.896181       262144       262144.0         5.0000         3.4334       23 h
    0.905591 0.870752       524288       524288.0         4.0000         3.9650       23 h
    0.881551 0.857512      1048576      1048576.0         5.0000         4.0165       23 h
    
    finished run
    number of examples per pass = 81513
    passes used = 20
    weighted example sum = 1630260.000000
    weighted label sum = 5743160.000000
    average loss = 0.835310 h
    best constant = 3.522849
    total feature number = 37495980
    awk: write error on stdout
     input record number 5598, file 
     source line number 1


```
creating quadratic features for pairs: ui
only testing
predictions = p_out
using no cache
Reading datafile = /dev/stdin
num sources = 1
Num weight bits = 18
learning rate = 0.015
initial_t = 1
power_t = 0
Enabled learners: rank, scorer-identity, count_label
Input label = SIMPLE
Output pred = SCALAR
average  since         example        example        current        current  current
loss     last          counter         weight          label        predict features

finished run
number of examples = 0
weighted example sum = 0.000000
weighted label sum = 0.000000
average loss = n.a.
total feature number = 0
➜  matrix-factorization git:(master) ✗ ./mf-run.sh
creating quadratic features for pairs: ui
using l2 regularization = 0.001
final_regressor = movielens.reg
using cache_file = movielens.cache
ignoring text input in favor of cache input
num sources = 1
Num weight bits = 18
learning rate = 0.015
initial_t = 1
power_t = 0
decay_learning_rate = 0.97
Enabled learners: rank, scorer-identity, count_label
Input label = SIMPLE
Output pred = SCALAR
average  since         example        example        current        current  current
loss     last          counter         weight          label        predict features
23.47829 23.47829            1            1.0         5.0000         0.1546       23
15.52144 7.564596            2            2.0         3.0000         0.2496       23
12.98887 10.45630            4            4.0         3.0000         0.3375       23
11.25910 9.529338            8            8.0         1.0000         0.6490       23
12.26689 13.27468           16           16.0         3.0000         1.1538       23
8.234041 4.201188           32           32.0         2.0000         1.6854       23
5.874644 3.515247           64           64.0         1.0000         2.7666       23
3.985317 2.095991          128          128.0         4.0000         3.1239       23
2.961340 1.937362          256          256.0         4.0000         2.3446       23
2.442762 1.924185          512          512.0         5.0000         2.9704       23
1.816145 1.189528         1024         1024.0         3.0000         3.6345       23
1.634253 1.452361         2048         2048.0         3.0000         3.9241       23
1.408795 1.183336         4096         4096.0         4.0000         3.6946       23
1.225044 1.041293         8192         8192.0         5.0000         3.7049       23
1.167822 1.110601        16384        16384.0         3.0000         3.2662       23
1.089689 1.011556        32768        32768.0         2.0000         3.8134       23
1.034541 0.979392        65536        65536.0         4.0000         4.1479       23
0.984682 0.984682       131072       131072.0         3.0000         3.5838       23 h
0.940430 0.896181       262144       262144.0         5.0000         3.4334       23 h
0.905591 0.870752       524288       524288.0         4.0000         3.9650       23 h
0.881551 0.857512      1048576      1048576.0         5.0000         4.0165       23 h

finished run
number of examples per pass = 81513
passes used = 20
weighted example sum = 1630260.000000
weighted label sum = 5743160.000000
average loss = 0.835310 h
best constant = 3.522849
total feature number = 37495980
```


```python
!./mf-test.sh
```

    creating quadratic features for pairs: ui
    predictions = p_out
    using no cache
    Reading datafile = test.data
    num sources = 1
    Num weight bits = 18
    learning rate = 0.015
    initial_t = 1
    power_t = 0
    Enabled learners: rank, scorer-identity, count_label
    Input label = SIMPLE
    Output pred = SCALAR
    average  since         example        example        current        current  current
    loss     last          counter         weight          label        predict features
    0.174680 0.174680            1            1.0         4.0000         3.5821       23
    0.373723 0.572766            2            2.0         4.0000         3.2432       23
    0.226993 0.080263            4            4.0         3.0000         3.4005       23
    0.464194 0.701394            8            8.0         3.0000         4.1566       23
    0.349510 0.234826           16           16.0         3.0000         3.1710       23
    0.914293 1.479076           32           32.0         4.0000         3.5122       23
    0.799798 0.685304           64           64.0         3.0000         4.3450       23
    0.744243 0.688687          128          128.0         4.0000         2.8742       23
    0.865488 0.986733          256          256.0         3.0000         2.7605       23
    0.900929 0.936370          512          512.0         4.0000         4.5798       23
    0.918205 0.935481         1024         1024.0         3.0000         3.2732       23
    0.957355 0.996505         2048         2048.0         4.0000         3.4607       23
    0.913432 0.869508         4096         4096.0         3.0000         3.3092       23
    0.911864 0.910296         8192         8192.0         2.0000         2.8665       23
    
    finished run
    number of examples = 9430
    weighted example sum = 9430.000000
    weighted label sum = 33833.000000
    average loss = 0.903003
    best constant = 3.587805
    total feature number = 216890



```python
!head p_out
```

    3.582053
    3.243187
    3.988498
    3.400492
    2.644075
    3.211900
    4.342837
    4.156624
    4.074239
    3.856281



```python
!head ./ml-100k/ua.test
```

    1	20	4	887431883
    1	33	4	878542699
    1	61	4	878542420
    1	117	3	874965739
    1	155	2	878542201
    1	160	4	875072547
    1	171	5	889751711
    1	189	3	888732928
    1	202	5	875072442
    1	265	4	878542441



```python
!head ./ml-100k/ua.base
```

    1	1	5	874965758
    1	2	3	876893171
    1	3	4	878542960
    1	4	3	876893119
    1	5	3	889751712
    1	6	5	887431973
    1	7	4	875071561
    1	8	1	875072484
    1	9	5	878543541
    1	10	3	875693118


The data consist of (user, item, rating, date) events, where ratings are given on an (integer) scale of 1 to 5.

We format it into a vowpalwabbit-friendly format, where `u` is short for `user`, and `i` is short for `item`. The first column is the `rating`.

```txt
5 |u 1 |i 1
3 |u 1 |i 2
4 |u 1 |i 3
3 |u 1 |i 4
3 |u 1 |i 5
5 |u 1 |i 6
4 |u 1 |i 7
1 |u 1 |i 8
5 |u 1 |i 9
3 |u 1 |i 10
```


```python
!echo '5 |u 1 |i 1' | poetry run python -m vowpalwabbit /dev/stdin -i ./ml-100k/movielens.reg -t -p p_out
```

    creating quadratic features for pairs: ui
    only testing
    predictions = p_out
    using no cache
    Reading datafile = /dev/stdin
    num sources = 1
    Num weight bits = 18
    learning rate = 0.015
    initial_t = 1
    power_t = 0
    Enabled learners: rank, scorer-identity, count_label
    Input label = SIMPLE
    Output pred = SCALAR
    average  since         example        example        current        current  current
    loss     last          counter         weight          label        predict features
    
    finished run
    number of examples = 0
    weighted example sum = 0.000000
    weighted label sum = 0.000000
    average loss = n.a.
    total feature number = 0



```python
!cat ml-100k/p_out
```


```python

```
