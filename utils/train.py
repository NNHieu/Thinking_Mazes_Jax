from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from mazes_data import DataLoader
from models import RecurModel
import optax
from functools import partial
import equinox as eqx
from tqdm import tqdm

def predict(net, iters, image):
  logits = net(image, iters=iters)
  return logits
  # Make a batched version of the `predict` function
batched_predict = jax.vmap(predict, in_axes=(None, None, 0))

def loss_fn(net, images, targets):
  logits = batched_predict(net, images).transpose(0, 2, 3, 1)
  targets_onehot = jax.nn.one_hot(targets, 2, axis=-1)
  return jnp.mean(optax.softmax_cross_entropy(logits, targets_onehot)), logits

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
  predicted_class = jnp.argmax(logits, axis=-1).astype(np.int32)
  return jnp.mean(jnp.amin(predicted_class == targets, axis=[-1, -2]))

class TrainState(NamedTuple):
  params: eqx.Module
  static: eqx.Module
  opt_state: optax.OptState
  tx: optax.GradientTransformation

@eqx.filter_jit
def train_step(state: TrainState, batch):
  params, static, opt_state, tx = state
  images, targets = batch
  net = eqx.combine(params[0], static)
  grad_fn = eqx.filter_value_and_grad(lambda net: loss_fn(net, images, targets), has_aux=True)
  (loss, logits), grads = grad_fn(net)
  updates, opt_state = tx.update((grads,), opt_state)
  params = optax.apply_updates(params, updates)
  metrics = {"loss": loss, "accuracy": accuracy(logits, targets)}
  state = state._replace(params=params, opt_state=opt_state)
  return state, metrics

@partial(jax.jit, static_argnums=(0,))
def eval_step(batched_predict, params, batch):
  images, targets = batch
  logits = batched_predict(params, images)
  loss  = optax.sigmoid_binary_cross_entropy(logits, targets)
  return {"loss": loss, "accuracy": accuracy(logits, targets)}

def ds_perms(batch_size, rng, ds):
  train_ds_size = len(ds[0])
  steps_per_epoch = train_ds_size // batch_size
  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  return perms

def data_generator(batch_size, rng, ds):
  perms = ds_perms(batch_size, rng, ds)
  for perm in tqdm(perms, leave=False):
    yield [v[perm, ...] for v in ds]
  

# @partial(jax.jit, static_argnums=(0, 1, 3))
def train_epoch(state: TrainState, dataloader: DataLoader, epoch):
    """Train for a single epoch."""
    batch_metrics = []
    for batch in tqdm(dataloader):
      state, metrics = train_step(state, batch)
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

# def create_train_state(rng, learning_rate, momentum):
#   """Creates initial `TrainState`."""
#   net = RecurModel(3, 128, key=rng)
#   params, static = eqx.partition(net, eqx.is_array)
#   tx = optax.adam(learning_rate)
#   opt_state = tx.init(params)
#   return (params, opt_state), tx, static
#   # tx = optax.adam(learning_rate)
#   # opt_state = tx.init(net)
#   # return (net, opt_state), tx, static

def create_train_state(rng, learning_rate, momentum):
  """Creates initial `TrainState`."""
  net = RecurModel(3, 128, key=rng)
  
  params, static = eqx.partition(net, eqx.is_array)
  params = (params,)
  recur_label = jax.tree_map(lambda _: 'nonrecur',  params)
  recur_label = eqx.tree_at(lambda params: params[0].iter_block, recur_label, replace='recur')
  # label_fn = lambda net: recur_mask
  # tx = optax.adam(learning_rate)
  tx = optax.multi_transform(
    {'recur': optax.adam(learning_rate*0.1), 'nonrecur': optax.adam(learning_rate)},
    recur_label
  )
  opt_state = tx.init(params)
  return TrainState(params, static, opt_state, tx)
  # tx = optax.adam(learning_rate)
  # opt_state = tx.init(net)
  # return (net, opt_state), tx, static