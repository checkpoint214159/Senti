"""
This module is meant to encapsulate the entire agent. 
Its scope contains all its components.
Its methods controls the flow of everything.

TODO the above is not true. it basically is the inference portion on its own, 
so later when you are designing the actual agent class that will also implement Planning,
convert this to inference? maybe? 
"""
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from torch import nn

from Senti.modules.agents.base import BaseAgent
from Senti.modules.config.config import ConfigDict
from Senti.modules.dataclass.action_policy import ActionPolicy
from Senti.modules.dataclass.normal import Normal
from Senti.modules.dataclass.obs import LatentObservation
from Senti.modules.dataclass.pomdpstate import POMDPState
from Senti.modules.optim.dist_loss import DistributionLosses
from Senti.modules.optim.optimregistry import OptimRegistry
from Senti.modules.policy.policy_predictor import PolicyPredictor
from Senti.modules.utils.caches import (
    Cache,
    CloneableCache,
    HierarchicalCache,
    StateCache,
)
from Senti.registry import ACTION_HEADS, AGENTS, AUTOENCODERS, DECODERS, WORLDMODELS


def set_seed(seed: int = 42):
    random.seed(seed)                  # Python random module
    np.random.seed(seed)               # NumPy
    torch.manual_seed(seed)            # CPU
    torch.cuda.manual_seed(seed)       # GPU
    torch.cuda.manual_seed_all(seed)   # all GPUs

    # For deterministic behavior (may be slower!)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Usage
set_seed(42)

ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)

def setup_logging(log_path: Path):
    """Set up logging only if not already configured."""
    if not logging.getLogger().hasHandlers():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

# Example usage
log_file = Path("logs/app.log")
setup_logging(log_file)

logging.info("Logging is set up!")

@AGENTS.register_module()
class Agent(BaseAgent):
    """
    An agent built on the concepts of active inference.
    This can be constructed either as an agent with its own cache of beliefs, optimizers, etc,
    or can be held as a mere module to encapsulate the various other things it uses.

    The former should be done during development, whilst treating it as a module
    renders the logic of 'what to do with agent modules' to a different component, like an AgentHandler,
    which is usually what is best for a 'full run', during experimentation and evaluation
    """

    def __init__(self,
        config: ConfigDict,
        as_module: bool = True,
        preference_genome = None,
    ):
        # configuration
        super().__init__(config, as_module)

        # encoder to encode incoming observation(s)
        self.obs_autoencoder = AUTOENCODERS.build("NmmoObsAE", self.config.NmmoObsAE)
        logging.info("Successfully loaded NmmoAE")
        # maps latent observation to state, for now only takes latent obs of image
        self.transition_model = WORLDMODELS.build("RSSM", self.config.transition_model)
        logging.info("Successfully loaded transition_model")
        self.grounding_wm = WORLDMODELS.build("RSSM", self.config.grounding_wm)
        logging.info("Successfully loaded grounding_wm")
        self.zh_o = DECODERS.build("LatentObsDecoder", self.config.zh_o)
        # self.preference_head = PREFERENCES.build("BasePreferenceHead", self.config.pref)
        self.habitual_head = ACTION_HEADS.build("ActionHead", self.config.habitual_head)
        self.deliberative_head = ACTION_HEADS.build("FiLMActionHead", self.config.deliberative_head)
        self.policy_predictor = PolicyPredictor(config=self.config.policy_predictor)

        self.atomic_timestep = self.config.atomic_timestep
        self.timestep_conv = nn.Conv1d(
            in_channels=self.atomic_timestep,
            out_channels=1,
            kernel_size=1
        )  # to aggregate across observational timesteps into one singular timestep 

        if self.as_module:
            self.__init_as_module__(preference_genome)

    def expose_genome(self) -> dict:
        """
        interface and exposes the contents of the agents deemed to be 
        """
        morphology = nn.ModuleDict({
            'transition_model': self.transition_model,
            'grounding_wm': self.grounding_wm,
            'deliberative_head': self.deliberative_head,
            'habitual_head': self.habitual_head,
            'obs_autoencoder': self.obs_autoencoder,
            'zh_o': self.zh_o,
            'timestep_conv': self.timestep_conv,
            'policy_predictor': self.policy_predictor,
        })


        return morphology
    
    def policy_init(self):
        # TODO: make it dependent on genome. for now just sample.
        return self.policy.sample()

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


    def update_latent_obs(self, timesteps:list[int], atomic_t:int):
        """
        TODO: assumes that the obs encoder operates on each timestep seperately, i.e it doesnt support us just
        stacking the tensor and doing one pass through. this should change
        """
        obs = []
        for t in timesteps:
            o = self.observation_cache.get(t)
            agent_embed_normal, others = self.obs_autoencoder.encode(o)
            obs.append(agent_embed_normal)

        stacked = torch.stack(obs, -2)
        atomic = self.timestep_conv(stacked)
        atomic = atomic.detach()   # VERY CRUCIAL TO PREVENT DOUBLE GRAD PROBLEM
        self._cache.set_container(
            'latent_obs', atomic_t, LatentObservation(lat_o=atomic.to(self.device)))


    def update_states_cache(self, atomic_t: int, obs: torch.Tensor):
        if atomic_t == 1:
            h = self.transition_model.initial_h(
                self.batch_size, 1  # magic number 1. i am so cooked. this is to represent timestep
            )  # TODO this could be learned?

        else:
            # calculate next h using previous state
            prev_state = self._cache.get_semantic('s', atomic_t - 1)
            # prev_action = self._cache.get_semantic('a', atomic_t - 1)
            prev_action = self._cache.get_semantic('a', atomic_t - 1)
            h, _ = self.transition_model.forward_h(prev_state, prev_action)

        z = self.transition_model.forward_z(h, external=obs)  # generate initial posterior
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


    def belief_optim_wrap(self):
        """
        Helper method to convert all states in state cache to parameters, wrapping around them with an optimizer
        """
        # seperate optimizer for over beliefs over states
        all_params = self._cache.get_params_flattened('states')
        kwargs = dict(lr=0.5)  # TODO fit this into config and organise optim registry call better
        self.optimizer_registry.register_optim('states_cache', all_params, optim_name='SGD', **kwargs)

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
            self.update_caches(timesteps, self.atomic_count)

            if (self.atomic_count % self.discrete_step) == 0 and self.atomic_count != 0:
                self.inference(
                    self.inference_steps,
                    self.param_learning_steps,
                )
            
                self.ground_wm()
                self.planning()
        # 6 - 9:
        
        # self.pi_head.predict('output_from_transmodel')

        # cleanup: shift atomic_window and remove all keys not in the atomic_window
        self.curr_timestep += 1
        self.prune()

    def inference(
        self,
        max_update_steps: int = 100,
        param_learning_steps: int = 10,
        print_statements: bool = True,
    ) -> float:
        """
        Run the inference procedure for belief updates and model parameter learning.
        Currently in the form of amortized inference, perhaps when I have time will fit in non-amortized (param learning
        only when VFE converges.)

        Args:
            max_update_steps (int): Maximum number of message passing steps to perform.
            param_learning_steps (int): Number of belief updates before checking VFE and (possibly) updating model parameters.
        Returns:
            float: Final VFE after inference and (optional) learning.
        """

        for step in range(max_update_steps):
            if step == 0:  # first step, percieve for latest prior and wrap in optim
                self.belief_optim_wrap()
                
            # random timestep group selection, to run inference on
            timestep = random.choice(list(self._cache.keys('states')))
            print('-----------Selected atomic timestep---------------', timestep)
            self.inference_step(timestep,
                # P_states,
                backprop_belief=True)  # lr scheduling comes later. test first
            self._cache.map_cache('latent_obs', lambda c: c.clone(detach=True))
            # self._cache.map_cache('api', lambda c: c.clone(detach=True))

            # every 'param_learning_steps', do param learning (backprop_belief=False)
            if step % param_learning_steps == 0 and step != 0:
                total_vfe = sum(
                    self.inference_step(
                        timestep=t,
                        # P_states=P_states,
                        backprop_belief=False
                    ) for t in self._cache.keys('states')
                )
                self._cache.map_cache('latent_obs', lambda c: c.clone(detach=True))
                print(f"[Step {step}] VFE = {total_vfe:.6f}") if print_statements else None

                self.optimizer_registry.do_step(
                    key='inference',
                    loss=total_vfe
                )

        return total_vfe


    def inference_step(
            self,
            timestep,
            backprop_belief=False
        ):
        """
        Step function to perform belief updating over cache of beliefs.

        Args:
            timestep: Selected timestep with which we perform inference for
            P_states: Dict of Ph_t 
            backprop_belief: Boolean on whether or not to call belief optimizer.
        """
        s_t: POMDPState = self._cache.get_semantic('s', timestep)

        # p(z_t | h_t) kl q(z_t | h_t, o_t)
        h_t = s_t.get('h')
        pz_t = self.transition_model.forward_z(h_t)

        qz_t = s_t.get('z')
        self.loss_module.include(
            ('qz_t vs pz_t', self.loss_module.kl,
            qz_t, pz_t))
            
        tm1 = timestep - 1
        if self._cache.has_semantic('s', tm1) and self._cache.has_semantic('a', tm1):
            s_tm1 = self._cache.get_semantic('s', tm1)
            a_tm1 = self._cache.get_semantic('a', tm1)
            # -------- how well does posterior from t-1 predict posterior of t -------- 
            _, _, pz_t_from_tm1= self.transition_model.forward(s_tm1, a_tm1)
            self.loss_module.include(
                ('qz_t vs tm1 -> pz_t', self.loss_module.kl,
                qz_t, pz_t_from_tm1))
            
            # ------- loss from action or something lol --------
            # TODO do we try and do this? will have to predict a from pz_t_from_tm1, and ph_t_from_tm1, which may be mroe unstable?
        
        tp1 = timestep + 1
        if self._cache.has_semantic('s', tp1) and self._cache.has_semantic('a', tp1):
            qz_tp1 = self._cache.get_semantic('z', tp1)
            a_tp1 = self._cache.get_semantic('a', tp1)
            _, _, pz_at_tp1= self.transition_model.forward(s_t, a_tp1)
            self.loss_module.include(
                ('qz_tp1 vs t -> pz_tp1', self.loss_module.kl,
                qz_tp1, pz_at_tp1))

        states_energy = self.loss_module.compute()

        # message from obs
        qz_t, qh_t = s_t.get('z').sample(), s_t.get('h').as_tensor()
        lat_o_pred = self.zh_o(qz_t, qh_t)
        lat_o = self._cache.get_semantic('lat_o', timestep)

        obs_MSE = self.loss_module.MSE(lat_o, lat_o_pred)  # positive equivalent for NLL if latent obs
        # was a variational belief. but it isnt, its just some tensors
        # it should help to update zh_o and qz_t and qh_t though
        
        total_energy = states_energy + obs_MSE
        # total_energy = states_energy

        if backprop_belief:
            self.optimizer_registry.do_step(
                key='states_cache',
                loss=total_energy
            )

        return total_energy

    def ground_wm(self):
        """
        Updates predictive WM to bootstrap it to transition model.
        Since during planning we operate entirely within imagination space, no observation semantics
        can be spotted here
        """
        # format x
        print('-----------------GROUNDING WM STEP-----------------')
        all_a = self._cache.get('api') \
            .vmap(lambda s: s.get('a')) \
            .values()
        all_z = self._cache.get('states') \
            .vmap(lambda s: s.get('z')) \
            .values()
        all_h = self._cache.get('states') \
            .vmap(lambda s: s.get('h')) \
            .values()
        
        all_z = type(all_z[0]).concat(all_z, 1)
        z = all_z.map(lambda z: z[:, :-1])  # exclude the final state when passing to wm

        # TODO This is horrible practice, hardcoding [:-1] like that 
        a = torch.concat(all_a[:-1], 1)
        seed_h = all_h[0]
        s = POMDPState(
            h=seed_h,
            z=z,  # [B, L-1, z_emb]
        )

        # forward
        pred_h, _, pred_z = self.grounding_wm.forward(s, a) # [B, L-1, h_emb]
        # correct wm to accurately predict our beliefs, based only off h.
        ground_z = all_z.map(lambda z: z[:, 1:])  # get all except first state
        self.loss_module.include(
            ('wm_grounding', self.loss_module.kl, 
            pred_z, ground_z),
        )

        pred_a = self.habitual_head.forward_hz(pred_h, pred_z)
        action_mse = self.loss_module.MSE(pred_a, a)

        grounding_loss = self.loss_module.compute() + action_mse

        self.optimizer_registry.do_step(
            key='wm',
            loss=grounding_loss)
        
    def save_genome(self):
        self.genome.save()

    # def load_genome(self):

    def planning(
        self,
    ):
        """
        do planning: sample n policies, run rollout function with it.
        Repeat this until EFE converges or we run out of steps

        Then, use this updated policy to predict action
        Use an affine transformation to augment the pre-sampling policy

        Start at t-1, because we dont have action for the current step, so
        we must take the previous action
        """
        # first, create planning priors from wm
        seed_s = self._cache.get_semantic('s', self.atomic_count - 1)
        seed_h, seed_z = seed_s.get('h'), seed_s.get('z')
        seed_a = self.habitual_head.forward_hz(seed_h, seed_z)

        s, a = seed_s, seed_a
        priors = [s]
        with torch.no_grad():
            # freeze gradients for prior_z
            for _ in range(self.planning_horizon):  # predict t+1, t+2, ...
                h, _, z = self.grounding_wm.forward(s, a)
                a = self.habitual_head.forward_hz(h, z)
                s = POMDPState(
                    h=h,
                    z=z,
                )
                priors.append(s)
            prior_z = [s.get('z') for s in priors]

        old_policy = self.policy.clone(detach=True)

        for i in range(self.planning_cycles):
            self.planning_step(seed_s, seed_a, prior_z)

        # lastly, sample updated policy
        final_pi = self.policy.sample()
        final_a = self.deliberative_head.forward_hz(
            seed_h, seed_z, self.genome, final_pi)
        self.prev_api = ActionPolicy(
            a=final_a.detach(),
            pi=final_pi.detach()
        )    

        # TODO merge new policy and old one tgt


    def planning_step(
        self,
        seed_s,
        seed_a,
        prior_z,
    ):  
        
        efe = 0
        for _ in range(self.n_policies):
            pi = self.policy.sample()
            rollout = self.rollout(seed_s=seed_s, seed_a=seed_a, policy=pi)
            rollout_z = [s.get('z') for s in rollout]
            [self.loss_module.include(
                ('instrumental_value', self.loss_module.kl, 
                rz, pz),
            ) for rz, pz in zip(rollout_z, prior_z)]
            instrumental = self.loss_module.compute()

            epistemic = sum([s.get('z').entropy() for s in rollout])
            # print('instrumental?', instrumental, 'epistemic', epistemic)
            efe += instrumental - epistemic

        self.optimizer_registry.do_step(
            key='policy',
            loss=efe
        )


    def rollout(
            self,
            seed_s,
            seed_a,
            policy,
        ):
        """
        Using the grounding wm as our prior, sample various policies and derive EFE for each one from rollouts
        """
        # create planning priors from wm.
        rollout = [seed_s]
        s, a = seed_s, seed_a
        # print('seed_s', seed_s)
        for _ in range(self.planning_horizon):
            h, _, z = self.transition_model.forward(s, a)
            a = self.deliberative_head.forward_hz(
                h, z, self.genome, policy
            )
            s = POMDPState(
                h=h,
                z=z,
            )
            rollout.append(s)

        return rollout


    
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


    def empty_action_init(self):
        """helper method to lazily create empty a."""
        return torch.zeros(self.batch_size, 1, self.a_dim,).to(self.device) # TODO fix the magic number. it really is supposed to be 1,
            # to represent the singular atomic timestep, but this is horrible practice.

    
    def __init_as_module__(self, preference_genome):   
        assert preference_genome is not None, """
        Assertion failed. If running this Agent as an individual module, \
        that is, it posesses and uses its own cache,  \
        you must pass in a preference genome tensor.
        """
        self.genome = preference_genome.to(self.device)

        # caches in observation space
        self.observation_cache = Cache(value_type=dict)
        self.action_cache = Cache(value_type=dict)

        # caches that respect atomic timestep
        self.latent_observation = CloneableCache(value_type=LatentObservation)
        self.action_policy = CloneableCache(value_type=ActionPolicy)
        self.states = StateCache(value_type=POMDPState)
        self._cache = HierarchicalCache(
            cache_names={
                'states': self.states,
                'latent_obs': self.latent_observation,
                'api': self.action_policy,
            },
            mapping={
                's': 'states',
                'h': 'states',
                'z': 'states',
                'lat_o': 'latent_obs',
                'a': 'api',
                'pi': 'api',
            },
        )

        # other non-model modules
        self.loss_module = DistributionLosses()
        self.optimizer_registry = OptimRegistry()

        self.policy: Normal = self.policy_predictor(preference_genome)

        # optimization
        inference_params = (
            list(self.obs_autoencoder.parameters()) + 
            list(self.transition_model.parameters()) +
            list(self.zh_o.parameters()) +
            list(self.deliberative_head.parameters())
        )
        kwargs = dict(lr=0.005)
        self.optimizer_registry.register_optim(
            'inference', inference_params, **kwargs
        )

        wm_params = (
            list(self.grounding_wm.parameters()) +
            list(self.habitual_head.parameters())
        )
        kwargs = dict(lr=0.005)
        self.optimizer_registry.register_optim(
            'wm', wm_params, **kwargs
        )

        policy_params = (
            list(self.policy.parameters())
        )
        kwargs = dict(lr=0.005)
        self.optimizer_registry.register_optim(
            'policy', policy_params, **kwargs
        )

        self.to(self.device)
        self.history = self.config.history  # this means when inference is run, NOT inclusive of latest obs, there are these many past tiemsteps
        self.device = torch.device(self.config.device)
        self.batch_size = self.config.batch_size
        self.inference_steps = self.config.inference_steps
        self.param_learning_steps = self.config.param_learning_steps
        self.planning_horizon = self.config.planning_horizon
        self.n_policies = self.config.n_policies
        self.planning_cycles = self.config.planning_cycles
        self.policy_dim = self.config.policy_dim
        
        self.discrete_step = self.config.discrete_step
        self.atomic_count = 0
        self.curr_timestep = 1
        self.curr_h_timestep = lambda: self.curr_h_timestep - 1  # additional variable to help with understanding.
        # for each node of state, we store h_t and z_t+1. this semantic is for convenience passing into the
        # world models, but it also makes sense because it predicts h_t+1 and z_t+2, which we can save into
        # a statenode once more.
        self.atomic_window = range(self.atomic_count - self.history, self.atomic_count)
        self.prev_api: ActionPolicy | None = None


# config = OmegaConf.load("/mnt/e/nmmo_actinf/Senti/src/Senti/config.yaml")
# agent = Agent(config)

# def unconvert(obj):
#     if isinstance(obj, dict):
#         return {k: unconvert(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return torch.from_numpy(np.asarray(obj).astype(np.float32)).to(device)
#     return obj

# for i in range(10):
#     response = requests.post(
#         'http://localhost:8000/take_step',
#         # json={
#         #     'action': noop_action,
#         # }
#     )

#     return_dict = unconvert(json.loads(response.content))
#     print('return_dict code', return_dict['view_code'])
#     print('return_dict tensor data', return_dict['data'].shape)

#     # torch_dtype = pufferlib.pytorch.nativize_dtype(puffer_env.emulated)
#     # torch_observation = pufferlib.pytorch.nativize_tensor(flat_torch_observation, torch_dtype)
        
#     # agent.forward(observations=return_dict)