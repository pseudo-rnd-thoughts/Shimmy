"""Wrapper to convert a meltingpot substrate into a pettingzoo compatible environment.

Taken from
https://github.com/deepmind/meltingpot/blob/main/examples/pettingzoo/utils.py
and modified to modern pettingzoo API
"""
# pyright: reportOptionalSubscript=false

from __future__ import annotations

import functools
from typing import Dict, Optional, Tuple

import gymnasium
import numpy as np
import pygame
from gymnasium.utils.ezpickle import EzPickle
from meltingpot.python import substrate
from ml_collections import config_dict
from pettingzoo.utils.env import ActionDict, AgentID, ObsDict, ParallelEnv

import shimmy.utils.meltingpot as utils


class MeltingPotCompatibilityV0(ParallelEnv, EzPickle):
    """This compatibility wrapper converts a meltingpot substrate into a pettingzoo environment.

    Melting Pot is a research tool developed to facilitate work on multi-agent artificial intelligence.
    It assesses generalization to novel social situations involving both familiar and unfamiliar individuals,
    and has been designed to test a broad range of social interactions such as: cooperation, competition,
    deception, reciprocation, trust, stubbornness and so on.
    Melting Pot offers researchers a set of over 50 multi-agent reinforcement learning substrates (multi-agent games)
    on which to train agents, and over 256 unique test scenarios on which to evaluate these trained agents.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    PLAYER_STR_FORMAT = "player_{index}"
    MAX_CYCLES = 1000

    def __init__(
        self, substrate_name: str, render_mode: str | None, max_cycles: int = MAX_CYCLES
    ):
        """Wrapper that converts a openspiel environment into a pettingzoo environment.

        Args:
            substrate_name (str): name of meltingpot substrate to load
            render_mode (Optional[str]): render_mode
            max_cycles (Optional[int]): maximum number of cycles (steps) before termination
        """
        # Create env config
        self.substrate_name = substrate_name
        self.player_roles = substrate.get_config(
            self.substrate_name
        ).default_player_roles
        self.env_config = {"substrate": self.substrate_name, "roles": self.player_roles}
        self.render_mode = render_mode
        self.max_cycles = max_cycles
        EzPickle.__init__(self, self.render_mode, self.env_config, self.max_cycles)

        # Build substrate from pickle
        self.env_config = config_dict.ConfigDict(self.env_config)
        self._env = substrate.build(
            self.env_config["substrate"], roles=self.env_config["roles"]
        )

        self.state_space = utils.dm_spec2gym_space(
            self._env.observation_spec()[0]["WORLD.RGB"]
        )

        # Set agents
        self._num_players = len(self._env.observation_spec())
        self.possible_agents = [
            self.PLAYER_STR_FORMAT.format(index=index)
            for index in range(self._num_players)
        ]
        self.agents = [agent for agent in self.possible_agents]

        # Set up pygame rendering
        if self.render_mode == "human":
            self.display_scale = 4
            self.display_fps = 5

            pygame.init()
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Melting Pot")
            shape = self.state_space.shape
            self.game_display = pygame.display.set_mode(
                (
                    int(shape[1] * self.display_scale),
                    int(shape[0] * self.display_scale),
                )
            )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """observation_space.

        Get the observation space from the underlying meltingpot substrate.

        Args:
            agent (AgentID): agent

        Returns:
            observation_space: spaces.Space
        """
        observation_space = utils.remove_world_observations_from_space(
            utils.dm_spec2gym_space(self._env.observation_spec()[0])  # type: ignore
        )
        return observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """action_space.

        Get the action space from the underlying meltingpot substrate.

        Args:
            agent (AgentID): agent

        Returns:
            action_space: spaces.Space
        """
        action_space = utils.dm_spec2gym_space(self._env.action_spec()[0])
        return action_space

    def state(self) -> np.ndarray:
        """State.

        Get an observation of the current environment's state. Used in rendering.

        Returns:
            observation
        """
        return self._env.observation()

    def reset(
        self,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None,
    ) -> ObsDict:
        """reset.

        Resets the environment.

        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with

        Returns:
            (observation, info)
        """
        timestep = self._env.reset()
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        return utils.timestep_to_observations(timestep)

    def step(
        self, actions: ActionDict
    ) -> tuple[
        ObsDict, dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]
    ]:
        """step.

        Updates the environment with an action.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        actions = [actions[agent] for agent in self.agents]
        timestep = self._env.step(actions)
        rewards = {
            agent: timestep.reward[index] for index, agent in enumerate(self.agents)
        }
        self.num_cycles += 1
        termination = timestep.last()
        terminations = {agent: termination for agent in self.agents}
        truncation = self.num_cycles >= self.max_cycles
        truncations = {agent: truncation for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        if termination or truncation:
            self.agents = []

        observations = utils.timestep_to_observations(timestep)
        return observations, rewards, terminations, truncations, infos

    def close(self):
        """close.

        Closes the environment.
        """
        self._env.close()

    def render(self) -> None | np.ndarray:
        """render.

        Renders the environment.

        Returns:
            The rendering of the environment, depending on the render mode
        """
        rgb_arr = self.state()[0]["WORLD.RGB"]
        if self.render_mode == "human":
            rgb_arr = np.transpose(rgb_arr, (1, 0, 2))
            surface = pygame.surfarray.make_surface(rgb_arr)
            rect = surface.get_rect()
            surf = pygame.transform.scale(
                surface,
                (int(rect[2] * self.display_scale), int(rect[3] * self.display_scale)),
            )

            self.game_display.blit(surf, dest=(0, 0))
            pygame.display.update()
            self.clock.tick(self.display_fps)
            return None
        elif self.render_mode == "pyplot":
            plt.cla()
            plt.imshow(rgb_arr, interpolation="nearest")
            plt.show(block=False)
            return None
        return rgb_arr
