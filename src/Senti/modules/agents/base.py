import torch
from torch import nn
from abc import ABC
from omegaconf.dictconfig import DictConfig
from Senti.modules.utils.caches import Cache

# TODO should we really make a base class this early?

class BaseActInfAgent(ABC, nn.Module):
    """
    abc class just to conform with a standard interface for the sake
    of our active inference research. We make not many assumptions beyond the idea that it will operate over
    discrete timesteps, and that the user may want to configure for an "atomic_timestep" which defines the timestep
    granularity in which inference and planning is performed.

    Also to provide inference and planning methods
    """
    def __init__(self,
        config: DictConfig,
    ):
        """
        Base Class args:
        """
        self.config = config
        self.history = self.config.history  # this means when inference is run, NOT inclusive of latest obs, there are these many past tiemsteps
        self.inference_steps = self.config.inference_steps
        self.atomic_timestep = config.atomic_timestep
        self.discrete_step = config.discrete_step
        self.atomic_count = 0
        self.curr_timestep = 1

        self.observation_cache = Cache(value_type=dict)

    def forward(self,
                observations: dict,
            ):
        """
        Order of execution:
            Inference:
                1. Encode current observation into a latent observation.
                2. Use it to predict z and h (h is always one timestep behind its representative timestep in the cache)
                    which act as our state
                3. From existing state beliefs in the belief cache, do belief updating
                4. Every n steps of belief updating, calculate total VFE and backprop model parameters
                5. Ground world-model to match our new beliefs
            Planning:
                6. Use WM to generate prior over future states
                7. Sample a policy (params) for action head, then do rollouts (using transition model) and calculate EFE
                8. Use EFE to update belief over policies
                9. Repeat until convergence, then sample policy and predict next action
        Then increment timestep by 1.
        """
        t = self.curr_timestep
        self.observation_cache.add(t, observations)

        if (self.curr_timestep % self.atomic_timestep) == 0:
            self.atomic_count += 1
            timesteps = list(range(self.curr_timestep - self.atomic_timestep + 1, self.curr_timestep + 1))
            self.update_latent_obs(timesteps, self.atomic_count)

            if (self.atomic_count % self.discrete_step) == 0 and self.atomic_count != 0:
                self.inference(self.inference_steps)
                self.planning()

        self.curr_timestep += 1
        self.prune()

