"""Utility functions for the compatibility wrappers."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import dm_env.specs
import numpy as np
import tree
from gymnasium import spaces


def dm_spec2gym_space(spec: tree.Structure[dm_env.specs.Array]) -> spaces.Space[Any]:
    """Converts a dm_env nested structure of specs to a Gymnasium Space.

    BoundedArray is converted to Box Gymnasium spaces. DiscreteArray is converted to
    Discrete Gymnasium spaces. Using Tuple and Dict spaces recursively as needed.

    Args:
      spec: The nested structure of specs

    Returns:
      The Gymnasium space corresponding to the given spec.
    """
    if isinstance(spec, (OrderedDict, dict)):
        return spaces.Dict(
            {key: dm_spec2gym_space(sub_spec) for key, sub_spec in spec.items()}
        )
    elif isinstance(spec, (list, tuple)):
        return spaces.Tuple([dm_spec2gym_space(sub_spec) for sub_spec in spec])
    # Due to inheritance we must use the Order - DiscreteArray, BoundedArray, StringArray, Array
    elif isinstance(spec, dm_env.specs.DiscreteArray):
        return spaces.Discrete(spec.maximum)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        return spaces.Box(
            low=spec.minimum, high=spec.maximum, shape=spec.shape, dtype=spec.dtype
        )
    elif isinstance(spec, dm_env.specs.StringArray):
        raise TypeError(spec)
    elif isinstance(spec, dm_env.specs.Array):
        raise TypeError(spec)
    else:
        raise NotImplementedError(
            f"Cannot convert dm_spec to gymnasium space, unknown spec: {spec}, please report."
        )


def dm_obs2gym_obs(obs) -> np.ndarray | dict[str, Any]:
    """Converts a dm_env observation to a gymnasium observation.

    Array observations are converted to numpy arrays. Dict observations are converted recursively per key.

    Args:
        obs: The dm_env observation

    Returns:
        The Gymnasium-compatible observation.
    """
    if isinstance(obs, (OrderedDict, dict)):
        return {key: dm_obs2gym_obs(value) for key, value in obs.items()}
    else:
        return np.asarray(obs)


def dm_env_step2gym_step(timestep) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    """Converts a dm_env timestep to the required return info from Gymnasium step() function.

    Args:
        timestep: The dm_env timestep

    Returns:
        observation, reward, terminated, truncated, info.
    """
    obs = dm_obs2gym_obs(timestep.observation)
    reward = timestep.reward or 0

    # set terminated and truncated
    terminated, truncated = False, False
    if timestep.last():
        # https://github.com/deepmind/dm_env/blob/master/docs/index.md#example-sequences
        if timestep.discount > 0:
            truncated = True
        else:
            terminated = True

    info = {
        "timestep.discount": timestep.discount,
        "timestep.step_type": timestep.step_type,
    }

    return (
        obs,
        reward,
        terminated,
        truncated,
        info,
    )
