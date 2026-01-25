import torch
from omegaconf.dictconfig import DictConfig
from torch import nn
from torch.func import functional_call, vmap
from torch.nn.utils.stateless import _reparametrize_module

from Senti.modules.agents.base import BaseAgent, BaseAgentHandler, GenotypeStrategy
from Senti.modules.agents.pomdp_agent import Agent
from Senti.modules.dataclass.action_policy import ActionPolicy
from Senti.modules.dataclass.normal import Normal
from Senti.modules.dataclass.obs import LatentObservation
from Senti.modules.dataclass.pomdpstate import POMDPState
from Senti.modules.genome.preference_genome import PreferenceGenome
from Senti.modules.optim.dist_loss import DistributionLosses
from Senti.modules.optim.optimregistry import OptimRegistry
from Senti.modules.policy.policy_predictor import PolicyPredictor
from Senti.modules.utils.caches import Cache, HierarchicalCache, create_POMDP_cache
from Senti.modules.utils.utils import nested_stack
from Senti.registry import AGENTS


@AGENTS.register_module()
class POMDPAgentHandler(BaseAgentHandler):

    def __init__(self,
        config: DictConfig,
    ):
        super().__init__(config)

        self.batch_size = self.config.batch_size
        self.history = self.config.history
        self.inference_steps = self.config.inference_steps
        self.atomic_timestep = self.config.atomic_timestep
        self.discrete_step = self.config.discrete_step
        self.atomic_count = 0
        self.curr_timestep = 1

        self.atomic_window = range(self.atomic_count - self.history, self.atomic_count)

        # caches in observation space
        observation_cache, action_cache, _cache = create_POMDP_cache()
        self.observation_cache: Cache = observation_cache
        self.action_cache: Cache = action_cache
        self._cache: HierarchicalCache = _cache
        self.prev_api: ActionPolicy | None = None
        
        self.state_dims = self.config.state_dims

    def load(self, strategy: dict | GenotypeStrategy):
        super().load(strategy)


    @classmethod
    def seed_preferences(self, pref_gene_dim: int, ids: list[int]):
        """
        Helper method to instantiate the preference genome
        This will be fed to the handler during the very first round of selection
        where our Selector is still empty.
        """
        return {
            id: PreferenceGenome(
                gene=nn.ParameterDict({'preferences': torch.nn.Parameter(torch.randn(pref_gene_dim))})
            ) for id in ids
        }


    def forward(self,
            observations: dict,
        ):

        t = self.curr_timestep
        self.observation_cache.add(t, observations)

        if (self.curr_timestep % self.atomic_timestep) == 0:
            self.atomic_count += 1
            timesteps = list(range(self.curr_timestep - self.atomic_timestep + 1, self.curr_timestep + 1))
            self.update_caches(timesteps, self.atomic_count)
            sdfg
            if (self.atomic_count % self.discrete_step) == 0 and self.atomic_count != 0:
                self.inference(
                    self.inference_steps,
                    self.param_learning_steps,
                )
            
                self.ground_wm()
                self.planning()
        
        self.curr_timestep += 1
        self.prune()

    def update_latent_obs(self, timesteps:list[int], atomic_t:int):
        """
        TODO: assumes that the obs encoder operates on each timestep seperately, i.e it doesnt support us just
        stacking the tensor and doing one pass through. this should change
        """
        obs_list = [self.observation_cache.get(t) for t in timesteps]
        obs_sequence = nested_stack(obs_list, dim=1)

        v_population = vmap(
            self.agent_blueprint.f_update_latent_obs,
            in_dims=(0, 0)
        )

        # detaches to prevent double grad problem
        population_atomic = v_population(self.merged_params, obs_sequence).detach()

        print('population_atomic shape?', population_atomic.shape)
        self._cache.set_container(
            'latent_obs', 
            atomic_t, 
            LatentObservation(lat_o=population_atomic.to(self.device))
        )

    def update_states_cache(self, atomic_t: int, obs: torch.Tensor):
        prev_state = self._cache.get_semantic('s', atomic_t - 1) if self._cache.has_semantic('s', atomic_t - 1) else None
        prev_action = self._cache.get_semantic('a', atomic_t - 1) if self._cache.has_semantic('a', atomic_t - 1) else None
        in_dims = [0, 0, 0, 0, None, None]

        # hardcode to change the None if prev_state / prev_action is None
        in_dims[2] = None if prev_state is None else 0
        in_dims[3] = None if prev_action is None else 0
        
        v_population = vmap(
            self.agent_blueprint.f_pred_new_state,
            in_dims=tuple(in_dims)
        )
        h, z = v_population(self.merged_params, obs, prev_state, prev_action,
            self.batch_size, atomic_t
        )
        h = h.clone(detach=True) # VERY CRUCIAL TO PREVENT DOUBLE GRAD PROBLEM
        z = z.clone(detach=True) # VERY CRUCIAL TO PREVENT DOUBLE GRAD PROBLEM

        self._cache.set_container('states', atomic_t, POMDPState(
            h=h,
            z=z).to(self.device)
        )

    def update_caches(self, timesteps:list[int], atomic_t:int):
        """
        calls various cache updating methods.
        """
        self.update_latent_obs(timesteps, atomic_t)
        if not self._cache.has_container('api', atomic_t):
            self.update_api_cache(atomic_t)  # TODO: decide to set here or immediately after planning.
            # here is neater, but technically if we freeze the agent after we make its move, it will have had
            # no idea what its last latent aciton was.
        if not self._cache.has_container('states', atomic_t):
            obs: torch.Tensor = self._cache.get_semantic('lat_o', atomic_t)
            self.update_states_cache(atomic_t, obs)

    def update_api_cache(self, atomic_t: int):
        if self.prev_api is None:
            a, pi = self.empty_action_init(), self.policy_init()
            prev_api = ActionPolicy(
                a=a,
                pi=pi,
            )
        else:
            prev_api = self.prev_api
        self._cache.set_container('api', atomic_t, prev_api)

    def prune(self):
        """
        Helper func to prune from caches if not in atomic_window.
        """
        new_atomic_window = range(self.atomic_count - self.history, self.atomic_count)
        removed_atomic_timesteps = list(set(self.atomic_window) - set(new_atomic_window))
        self.atomic_window = new_atomic_window
        interleaved = [
            sub_step 
            for x in removed_atomic_timesteps
            for sub_step in range((x - 1) * self.atomic_timestep + 1, x * self.atomic_timestep + 1)
        ]

        # from removed timesteps, just assume that for each element, subtract atomic_timesteps number
        # to get the actual timesteps to remove from obs cache.

        print('REMOVED ATOMIC TIMESTEPS', removed_atomic_timesteps)

        for t in removed_atomic_timesteps:
            self._cache.remove_container('states', t) \
                if self._cache.has_container('states', t) else None
            self._cache.remove_container('api', t) \
                if self._cache.has_container('api', t) else None
            self._cache.remove_container('latent_obs', t) \
                if self._cache.has_container('latent_obs', t) else None

        for t in interleaved:
            self.observation_cache.remove(t) if self.observation_cache.has(t) else None

    def empty_action_init(self) -> torch.Tensor:
        """helper method to sloppily create empty a."""
        return torch.zeros(self.batch_size, 1, self.state_dims.a_dim,).to(self.device) # TODO fix the magic number. it really is supposed to be 1,
        # to represent the singular atomic timestep, but this is horrible practice.


    def policy_init(self) -> torch.Tensor:
        # TODO: make it dependent on genome. for now just sample.
        v_population = vmap(
            self.agent_blueprint.f_predict_policy,
            in_dims=(0,),
            randomness='same',
        )

        # detaches to prevent double grad problem
        return v_population(self.merged_params)

