import jax
import jax.numpy as jnp

def nesterov_weight_decay(mass: float, weight_decay: float):
  """Construct optimizer triple for SGD with Nesterov momentum.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.
    mass: positive scalar representing the momentum coefficient.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  def init(x0):
    v0 = jnp.zeros_like(x0)
    return v0
  def update(step_size, g, x, velocity):
    g = g + weight_decay * x
    velocity = mass * velocity + g
    x = x - step_size * (mass * velocity + g)
    return x, velocity
  def get_params(state):
    return state[0]
  def get_velocity(state):
    return state[1]
  return init, update, get_params, get_velocity

