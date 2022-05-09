from asyncio import FastChildWatcher
from mazes_data import MazeDataset, prepare_maze_loader
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from models import RecurModel
import optax
from functools import partial
import equinox as eqx
from tqdm import tqdm

def predict(net, image):
  logits = net(image, iters=1)
  return logits
  # Make a batched version of the `predict` function
batched_predict = jax.vmap(predict, in_axes=(None, 0))

def loss_fn(net, images, targets):
  logits = batched_predict(net, images)
  return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, targets)), logits

# def predict(static, params, image):
#   net = eqx.combine(static, params)
#   logits = net(image, iters=30)
#   return logits
#   # Make a batched version of the `predict` function
# batched_predict = jax.vmap(predict, in_axes=(None, None, 0))

# def loss_fn(static, params, images, targets):
#   logits = batched_predict(static, params, images)
#   return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, targets)), logits

def accuracy(logits, targets):
  predicted_class = (logits > 0).astype(np.int32)
  return jnp.mean(jnp.amin(predicted_class == targets, axis=[-1, -2]))

@partial(jax.jit, static_argnums=(0,1))
def train_step(tx, loss_fn, state, batch):
  params, opt_state = state
  images, targets = batch
  grad_fn = eqx.filter_value_and_grad(lambda params: loss_fn(params, images, targets), has_aux=True)
#   def loss_fn1(params):
#     return loss_fn(params, images, targets)
#   grad_fn = jax.value_and_grad(lambda params: loss_fn(params, images, targets), has_aux=True)
  (loss, logits), grads = grad_fn(params)
  updates, opt_state = tx.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  metrics = {"loss": loss, "accuracy": accuracy(logits, targets)}
  state = (params, opt_state)
  return state, metrics

@partial(jax.jit, static_argnums=(0,))
def eval_step(batched_predict, params, batch):
  images, targets = batch
  logits = batched_predict(params, images)
  loss  = optax.sigmoid_binary_cross_entropy(logits, targets)
  return {"loss": loss, "accuracy": accuracy(logits, targets)}

# @partial(jax.jit, static_argnums=(0, 1, 3))
def train_epoch(tx, loss_fn, state, train_ds, batch_size, epoch, rng):
    """Train for a single epoch."""
    # batch_metrics = []
    train_ds_size = len(train_ds[0])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in tqdm(perms, leave=False):
        batch = [v[perm, ...] for v in train_ds]
        state, metrics = train_step(tx,loss_fn, state, batch)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}

    print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
        epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

    return state

def eval_model(batched_predict, params, val_loaders):
    batch_metrics = []
    for batch in tqdm(val_loaders, leave=False):
        metrics = eval_step(batched_predict, params, batch)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]}
    return epoch_metrics_np['loss'], epoch_metrics_np['accuracy']

def create_train_state(rng, learning_rate, momentum):
  """Creates initial `TrainState`."""
  net = RecurModel(3, 128, key=rng)
#   params, static = eqx.partition(net, eqx.is_array)

  tx = optax.adam(learning_rate)
  opt_state = tx.init(net)
  return (net, opt_state), tx, None