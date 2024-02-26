import pickle
import jax

class Checkpointer:
  def __init__(self, path):
    self.path = path

  def save(self, params):
    params = jax.device_get(params)
    with open(self.path, 'wb') as fp:
      pickle.dump(params, fp)

  def load(self):
    with open(self.path, 'rb') as fp:
      params = pickle.load(fp)
    return jax.device_put(params)
