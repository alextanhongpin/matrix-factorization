# Recommendation Systems with TensorFlow

NOTE: This example does not work. The code is still using the legacy implementation.


https://colab.research.google.com/github/google/eng-edu/blob/main/ml/recommendation-systems/recommendation-systems.ipynb?utm_source=ss-recommendation-systems&utm_campaign=colab-external&utm_medium=referral&utm_content=recommendation-systems#scrollTo=6NHoOwido4tk


```python
import collections

import matplotlib.pyplot as plt
import tensorflow as tf

from movie_helper.rating import load_ratings
```


```python
ratings = load_ratings()
ratings.head()
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
      <th>user_id</th>
      <th>item_id</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3</td>
      <td>881250949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>186</td>
      <td>302</td>
      <td>3</td>
      <td>891717742</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>377</td>
      <td>1</td>
      <td>878887116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>244</td>
      <td>51</td>
      <td>2</td>
      <td>880606923</td>
    </tr>
    <tr>
      <th>4</th>
      <td>166</td>
      <td>346</td>
      <td>1</td>
      <td>886397596</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings["user_id"] -= 1
ratings["item_id"] -= 1
```


```python
ratings.head()
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
      <th>user_id</th>
      <th>item_id</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>195</td>
      <td>241</td>
      <td>3</td>
      <td>881250949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>185</td>
      <td>301</td>
      <td>3</td>
      <td>891717742</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>376</td>
      <td>1</td>
      <td>878887116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>243</td>
      <td>50</td>
      <td>2</td>
      <td>880606923</td>
    </tr>
    <tr>
      <th>4</th>
      <td>165</td>
      <td>345</td>
      <td>1</td>
      <td>886397596</td>
    </tr>
  </tbody>
</table>
</div>




```python
(n_user,) = ratings["user_id"].unique().shape
(n_movie,) = ratings["item_id"].unique().shape
```


```python
def build_rating_sparse_tensor(ratings_df):
    indices = ratings_df[["user_id", "item_id"]].values
    values = ratings_df["rating"].values

    return tf.SparseTensor(
        indices=indices, values=values, dense_shape=[n_user, n_movie]
    )
```


```python
def sparse_mean_square_error(sparse_ratings, user_embeddings, movie_embeddings):
    predictions = tf.gather_nd(
        tf.matmul(user_embeddings, movie_embeddings, transpose_b=True),
        sparse_ratings.indices,
    )
    loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
    return loss
```


```python
def sparse_mean_square_error(sparse_ratings, user_embeddings, movie_embeddings):
    predictions = tf.reduce_sum(
        tf.gather(user_embeddings, sparse_ratings.indices[:, 0])
        * tf.gather(movie_embeddings, sparse_ratings.indices[:, 1]),
        axis=1,
    )
    loss_fn = tf.keras.losses.MeanSquaredError()
    loss = loss_fn(y_true=sparse_ratings.values, y_pred=predictions)
    return loss
```


```python
class CFModel(object):
    """Simple class that represents a collaborative filtering model"""

    def __init__(self, embedding_vars, loss, metrics=None):
        """Initializes a CFModel.
        Args:
          embedding_vars: A dictionary of tf.Variables.
          loss: A float Tensor. The loss to optimize.
          metrics: optional list of dictionaries of Tensors. The metrics in each
            dictionary will be plotted in a separate figure during training.
        """
        self._embedding_vars = embedding_vars
        self._loss = loss
        self._metrics = metrics
        self._embeddings = {k: None for k in embedding_vars}

    @property
    def embeddings(self):
        """The embeddings dictionary."""
        return self._embeddings

    def train(
        self,
        num_iterations=100,
        learning_rate=1.0,
        plot_results=True,
        optimizer=tf.keras.optimizers.SGD,
    ):
        """Trains the model.
        Args:
          iterations: number of iterations to run.
          learning_rate: optimizer learning rate.
          plot_results: whether to plot the results at the end of training.
          optimizer: the optimizer to use. Default to SGD.
        Returns:
          The metrics dictionary evaluated at the last iteration.
        """
        opt = optimizer(learning_rate)

        iterations = []
        metrics = self._metrics or ({},)
        metrics_vals = [collections.defaultdict(list) for _ in self._metrics]

        # Train and append results.
        for i in range(num_iterations + 1):

            with tf.GradientTape() as tape:
                train_op = opt.minimize(
                    self._loss, var_list=self._metrics, tape=tape
                )
            train_op()
            results = {name: metric.result() for name, metric in metrics.items()}
            if (i % 10 == 0) or i == num_iterations:
                print(
                    "\r iteration %d: " % i
                    + ", ".join(
                        ["%s=%f" % (k, v) for r in results for k, v in r.items()]
                    ),
                    end="",
                )
                iterations.append(i)
                for metric_val, result in zip(metrics_vals, results):
                    for k, v in result.items():
                        metric_val[k].append(v)

        for k, v in self._embedding_vars.items():
            self._embeddings[k] = v.numpy()

        if plot_results:
            # Plot the metrics.
            num_subplots = len(metrics) + 1
            fig = plt.figure()
            fig.set_size_inches(num_subplots * 10, 8)
            for i, metric_vals in enumerate(metrics_vals):
                ax = fig.add_subplot(1, num_subplots, i + 1)
                for k, v in metric_vals.items():
                    ax.plot(iterations, v, label=k)
                ax.set_xlim([1, num_iterations])
                ax.legend()
        return results
```


```python
def split_dataframe(df, holdout_fraction=0.1):
    """Splits a DataFrame into training and test sets.
    Args:
      df: a dataframe.
      holdout_fraction: fraction of dataframe rows to use in the test set.
    Returns:
      train: dataframe for training
      test: dataframe for testing
    """
    test = df.sample(frac=holdout_fraction, replace=False)
    train = df[~df.index.isin(test.index)]
    return train, test
```


```python
def build_model(ratings, embedding_dim=3, init_stddev=1.0):
    """
    Args:
      ratings: a DataFrame of the ratings
      embedding_dim: the dimension of the embedding vectors.
      init_stddev: float, the standard deviation of the random initial embeddings.
    Returns:
      model: a CFModel.
    """
    # Split the ratings DataFrame into train and test.
    train_ratings, test_ratings = split_dataframe(ratings)
    # SparseTensor representation of the train and test datasets.
    A_train = build_rating_sparse_tensor(train_ratings)
    A_test = build_rating_sparse_tensor(test_ratings)
    # Initialize the embeddings using a normal distribution.
    U = tf.Variable(
        tf.random.normal([A_train.dense_shape[0], embedding_dim], stddev=init_stddev)
    )
    V = tf.Variable(
        tf.random.normal([A_train.dense_shape[1], embedding_dim], stddev=init_stddev)
    )
    train_loss = sparse_mean_square_error(A_train, U, V)
    test_loss = sparse_mean_square_error(A_test, U, V)
    metrics = {"train_error": train_loss, "test_error": test_loss}
    embeddings = {"user_id": U, "movie_id": V}
    print("train_loss", train_loss)
    print("test_loss", test_loss)
    return CFModel(embeddings, train_loss, [metrics])
```


```python
# Build the CF model and train it.
model = build_model(ratings, embedding_dim=30, init_stddev=0.5)
model.train(num_iterations=1000, learning_rate=10.0)
```

    train_loss tf.Tensor(15.585228, shape=(), dtype=float32)
    test_loss tf.Tensor(15.7773285, shape=(), dtype=float32)



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[409], line 3
          1 # Build the CF model and train it.
          2 model = build_model(ratings, embedding_dim=30, init_stddev=0.5)
    ----> 3 model.train(num_iterations=1000, learning_rate=10.0)


    Cell In[406], line 48, in CFModel.train(self, num_iterations, learning_rate, plot_results, optimizer)
         45 for i in range(num_iterations + 1):
         47     with tf.GradientTape() as tape:
    ---> 48         train_op = opt.minimize(
         49             self._loss, var_list=self._metrics, tape=tape
         50         )
         51     train_op()
         52     results = {name: metric.result() for name, metric in metrics.items()}


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/keras/src/optimizers/optimizer.py:544, in _BaseOptimizer.minimize(self, loss, var_list, tape)
        523 """Minimize `loss` by updating `var_list`.
        524 
        525 This method simply computes gradient using `tf.GradientTape` and calls
       (...)
        541   None
        542 """
        543 grads_and_vars = self.compute_gradients(loss, var_list, tape)
    --> 544 self.apply_gradients(grads_and_vars)


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/keras/src/optimizers/optimizer.py:1222, in Optimizer.apply_gradients(self, grads_and_vars, name, skip_gradients_aggregation, **kwargs)
       1218 experimental_aggregate_gradients = kwargs.pop(
       1219     "experimental_aggregate_gradients", True
       1220 )
       1221 if not skip_gradients_aggregation and experimental_aggregate_gradients:
    -> 1222     grads_and_vars = self.aggregate_gradients(grads_and_vars)
       1223 return super().apply_gradients(grads_and_vars, name=name)


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/keras/src/optimizers/optimizer.py:1184, in Optimizer.aggregate_gradients(self, grads_and_vars)
       1182     return grads_and_vars
       1183 else:
    -> 1184     return optimizer_utils.all_reduce_sum_gradients(grads_and_vars)


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/keras/src/optimizers/utils.py:37, in all_reduce_sum_gradients(grads_and_vars)
         35 if tf.__internal__.distribute.strategy_supports_no_merge_call():
         36     grads = [pair[0] for pair in filtered_grads_and_vars]
    ---> 37     reduced = tf.distribute.get_replica_context().all_reduce(
         38         tf.distribute.ReduceOp.SUM, grads
         39     )
         40 else:
         41     # TODO(b/183257003): Remove this branch
         42     reduced = tf.distribute.get_replica_context().merge_call(
         43         _all_reduce_sum_fn, args=(filtered_grads_and_vars,)
         44     )


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/tensorflow/python/distribute/distribute_lib.py:3649, in ReplicaContextBase.all_reduce(self, reduce_op, value, options)
       3647     # The gradient of an all-sum is itself an all-sum (all-mean, likewise).
       3648     return ys, lambda *dy_s: self.all_reduce(reduce_op, dy_s)
    -> 3649   return nest.pack_sequence_as(value, grad_wrapper(*flattened_value))
       3650 else:
       3651   if has_indexed_slices:


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/tensorflow/python/ops/custom_gradient.py:343, in Bind.__call__(self, *a, **k)
        342 def __call__(self, *a, **k):
    --> 343   return self._d(self._f, a, k)


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/tensorflow/python/ops/custom_gradient.py:297, in custom_gradient.<locals>.decorated(wrapped, args, kwargs)
        295 """Decorated function with custom gradient."""
        296 if context.executing_eagerly():
    --> 297   return _eager_mode_decorator(wrapped, args, kwargs)
        298 else:
        299   return _graph_mode_decorator(wrapped, args, kwargs)


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/tensorflow/python/ops/custom_gradient.py:566, in _eager_mode_decorator(f, args, kwargs)
        563 flat_result = composite_tensor_gradient.get_flat_tensors_for_gradients(
        564     nest.flatten(result))
        565 # TODO(apassos) consider removing the identity below.
    --> 566 flat_result = [gen_array_ops.identity(x) for x in flat_result]
        568 input_tensors = [
        569     ops.convert_to_tensor(x) for x in flat_args + list(variables)]
        571 recorded_inputs = input_tensors


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/tensorflow/python/ops/custom_gradient.py:566, in <listcomp>(.0)
        563 flat_result = composite_tensor_gradient.get_flat_tensors_for_gradients(
        564     nest.flatten(result))
        565 # TODO(apassos) consider removing the identity below.
    --> 566 flat_result = [gen_array_ops.identity(x) for x in flat_result]
        568 input_tensors = [
        569     ops.convert_to_tensor(x) for x in flat_args + list(variables)]
        571 recorded_inputs = input_tensors


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/tensorflow/python/ops/gen_array_ops.py:4192, in identity(input, name)
       4190   pass
       4191 try:
    -> 4192   return identity_eager_fallback(
       4193       input, name=name, ctx=_ctx)
       4194 except _core._SymbolicException:
       4195   pass  # Add nodes to the TensorFlow graph.


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/tensorflow/python/ops/gen_array_ops.py:4212, in identity_eager_fallback(input, name, ctx)
       4211 def identity_eager_fallback(input: Annotated[Any, TV_Identity_T], name, ctx) -> Annotated[Any, TV_Identity_T]:
    -> 4212   _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
       4213   _inputs_flat = [input]
       4214   _attrs = ("T", _attr_T)


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/tensorflow/python/eager/execute.py:259, in args_to_matching_eager(***failed resolving arguments***)
        256     tensor = None
        258 if tensor is None:
    --> 259   tensor = tensor_conversion_registry.convert(
        260       t, dtype, preferred_dtype=default_dtype
        261   )
        263 ret.append(tensor)
        264 if dtype is None:


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/tensorflow/python/framework/tensor_conversion_registry.py:234, in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
        225       raise RuntimeError(
        226           _add_error_prefix(
        227               f"Conversion function {conversion_func!r} for type "
       (...)
        230               f"actual = {ret.dtype.base_dtype.name}",
        231               name=name))
        233 if ret is None:
    --> 234   ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
        236 if ret is NotImplemented:
        237   continue


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/tensorflow/python/framework/constant_op.py:335, in _constant_tensor_conversion_function(v, dtype, name, as_ref)
        332 def _constant_tensor_conversion_function(v, dtype=None, name=None,
        333                                          as_ref=False):
        334   _ = as_ref
    --> 335   return constant(v, dtype=dtype, name=name)


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/tensorflow/python/ops/weak_tensor_ops.py:142, in weak_tensor_binary_op_wrapper.<locals>.wrapper(*args, **kwargs)
        140 def wrapper(*args, **kwargs):
        141   if not ops.is_auto_dtype_conversion_enabled():
    --> 142     return op(*args, **kwargs)
        143   bound_arguments = signature.bind(*args, **kwargs)
        144   bound_arguments.apply_defaults()


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/tensorflow/python/framework/constant_op.py:271, in constant(value, dtype, shape, name)
        172 @tf_export("constant", v1=[])
        173 def constant(
        174     value, dtype=None, shape=None, name="Const"
        175 ) -> Union[ops.Operation, ops._EagerTensorBase]:
        176   """Creates a constant tensor from a tensor-like object.
        177 
        178   Note: All eager `tf.Tensor` values are immutable (in contrast to
       (...)
        269     ValueError: if called on a symbolic tensor.
        270   """
    --> 271   return _constant_impl(value, dtype, shape, name, verify_shape=False,
        272                         allow_broadcast=True)


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/tensorflow/python/framework/constant_op.py:284, in _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast)
        282     with trace.Trace("tf.constant"):
        283       return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    --> 284   return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
        286 const_tensor = ops._create_graph_constant(  # pylint: disable=protected-access
        287     value, dtype, shape, name, verify_shape, allow_broadcast
        288 )
        289 return const_tensor


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/tensorflow/python/framework/constant_op.py:296, in _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
        292 def _constant_eager_impl(
        293     ctx, value, dtype, shape, verify_shape
        294 ) -> ops._EagerTensorBase:
        295   """Creates a constant on the current device."""
    --> 296   t = convert_to_eager_tensor(value, ctx, dtype)
        297   if shape is None:
        298     return t


    File ~/Documents/python/matrix-factorization/.venv/lib/python3.10/site-packages/tensorflow/python/framework/constant_op.py:103, in convert_to_eager_tensor(value, ctx, dtype)
        101     dtype = dtypes.as_dtype(dtype).as_datatype_enum
        102 ctx.ensure_initialized()
    --> 103 return ops.EagerTensor(value, ctx.device_name, dtype)


    ValueError: Attempt to convert a value (None) with an unsupported type (<class 'NoneType'>) to a Tensor.



```python

```


```python

```
