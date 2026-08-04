from typing import Any, Callable, Dict, Sequence, Union

import flax
import jax.numpy as jnp
import numpy as np

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]
# tf.Tensor dropped from this union -- tensorflow is a heavy, unused dependency
# for the JAX-only training paths (BC, SAC/RLPD) actually exercised here; it
# was only ever needed by train_utils.load_recorded_video's wandb video logging.
Array = Union[np.ndarray, jnp.ndarray]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
# A method to be passed into TrainState.__call__
ModuleMethod = Union[str, Callable, None]
