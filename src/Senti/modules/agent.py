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
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch import nn

from Senti.modules.dataclasses.normal import Normal
from Senti.modules.dataclasses.pomdpstate import POMDPState
from Senti.modules.optim.dist_loss import DistributionLosses
from Senti.modules.optim.optimregistry import OptimRegistry
from Senti.modules.utils.caches import Cache, StateTimestepCache, TensorCache
from Senti.modules.utils.utils import nmmo_agent_check_config
from Senti.registry import AUTOENCODERS, WORLDMODELS

OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}))

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

class Agent(nn.Module):
    """
    An agent built on the concepts of active inference.
    """

    def __init__(self,
        config: DictConfig,
    ):
        """

        """
        # admin stuff
        super().__init__()
        self.config = config
        self.history = self.config.history  # this means when inference is run, NOT inclusive of latest obs, there are these many past tiemsteps
        self.device = torch.device(self.config.device)
        self.batch_size = self.config.batch_size
        self.inference_steps = self.config.inference_steps
        self.param_learning_steps = self.config.param_learning_steps
        self.planning_horizon = self.config.planning_horizon
        nmmo_agent_check_config(self.config)
        
        # observations and states cache.
        self.states_cache = StateTimestepCache(value_type=POMDPState)
        self.observation_cache = Cache(value_type=dict)
        self.latent_observation_cache = StateTimestepCache(value_type=Normal)
        self.action_cache = TensorCache()
        self.latent_action_cache = TensorCache()

        # dimensionality and tensor shapes
        self.state_depth = self.config.state_depth
        self.a_dim = self.z_dim  # TODO move to action head

        # other non-model modules
        self.loss_module = DistributionLosses()
        self.optimizer_registry = OptimRegistry()

        # encoder to encode incoming observation(s)
        self.obs_autoencoder = AUTOENCODERS.build("NmmoObsAE", config.NmmoObsAE)
        logging.info("Successfully loaded NmmoAE")
        self.z_ae = AUTOENCODERS.build("zAE", config.zAE)

        # maps latent observation to state, for now only takes latent obs of image
        self.transition_model = WORLDMODELS.build("RSSM", config.transition_model)
        logging.info("Successfully loaded transition_model")

        self.grounding_wm = WORLDMODELS.build("RSSM", config.grounding_wm)
        logging.info("Successfully loaded grounding_wm")

        self.policy = Normal(
            z_mean_shape=(self.config.preferences_dim,),
            z_log_std_shape=(self.config.preferences_dim,)
        )

        # optimization
        inference_params = (
            list(self.obs_autoencoder.parameters()) + 
            list(self.transition_model.parameters())
        )
        kwargs = dict(lr=0.005)
        self.optimizer_registry.register_optim(
            'inference', inference_params, **kwargs
        )

        wm_params = list(self.grounding_wm.parameters())
        kwargs = dict(lr=0.005)
        self.optimizer_registry.register_optim(
            'wm', wm_params, **kwargs
        )

        self.atomic_timestep = config.atomic_timestep
        self.discrete_step = config.discrete_step
        self.atomic_count = 0
        self.curr_timestep = 1
        self.curr_h_timestep = lambda: self.curr_h_timestep - 1  # additional variable to help with understanding.
        # for each node of state, we store h_t and z_t+1. this semantic is for convenience passing into the
        # world models, but it also makes sense because it predicts h_t+1 and z_t+2, which we can save into
        # a statenode once more.
        self.atomic_window = range(self.atomic_count - self.history, self.atomic_count)

        self.to(self.device)

    def expose_morphology(self) -> dict:
        """
        interface and exposes the contents of the agents deemed to be morphologically relevant
        """
        morphology = nn.ModuleDict({
            'transition_model': self.transition_model,
            'grounding_wm': self.grounding_wm,
            'obs_autoencoder': self.obs_autoencoder,
            'z_ae': self.z_ae
        })

        return morphology.state_dict()
    
    def expose_policy(self) -> dict:
        return {
            'mean': self.policy.mean,
            'log_std': self.policy.log_std
        }

    def initialize_state_node(self, atomic_t: int):
        a = self.empty_action_init()  # FOR NOW
        if atomic_t == 1:
            h = self.transition_model.initial_h()  # TODO this could be learned?
            # a = self.empty_action_init()
     
        else:
            # calculate next h using previous state
            state = self.states_cache.get(atomic_t - 1)
            h = self.transition_model.forward_h(state)
            # a, h = s.get('a'), s.get('h')

        z = self.forward_z(h, o_embed_t=None)  # generate prior for z here
   
        self.states_cache.add(atomic_t, POMDPState(
            h=h,
            z=z,
            a=a).to(self.device)
        )
        

    def update_latent_obs(self, timesteps:list[int], atomic_num:int):
        """
        TODO: assumes that the obs encoder operates on each timestep seperately, i.e it doesnt support us just
        stacking the tensor and doing one pass through. this should change
        """
        obs = []
        for t in timesteps:
            o = self.observation_cache.get(t)
            agent_embed_normal, others = self.obs_autoencoder.encode(o)
            obs.append(agent_embed_normal)

        stacked = type(obs[0]).concat_timesteps(obs)
        self.latent_observation_cache.add(
                atomic_num, stacked.to(self.device))
        

    def update_states_cache(self):
        """
        Simple helper method to update all timesteps in observation cache.
        """
        for atomic_t in self.latent_observation_cache.keys():
            if not self.states_cache.has(atomic_t):
                # obs: Normal = self.latent_observation_cache.get(atomic_t)
                self.initialize_state_node(atomic_t)
                

    def belief_optim_wrap(self):
        """
        Helper method to convert all states in state cache to parameters, wrapping around them with an optimizer
        """
        # seperate optimizer for over beliefs over states
        all_params = self.states_cache.get_params_flattened()
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
            self.update_latent_obs(timesteps, self.atomic_count)

            if (self.atomic_count % self.discrete_step) == 0 and self.atomic_count != 0:
                self.inference(
                    self.inference_steps,
                    self.param_learning_steps,
                )
                # self.ground_wm()

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
                self.update_states_cache()
                self.belief_optim_wrap()
                P_states = self.states_cache.replica(detach=True, freeze=True)

            # random timestep group selection, to run inference on
            timestep = random.choice(list(self.states_cache.keys()))
            print('-----------Selected atomic timestep---------------', timestep)
            self.inference_step(timestep, P_states, backprop_belief=True)  # lr scheduling comes later. test first
            P_states = P_states.replica(detach=True, freeze=True)
            self.latent_observation_cache = self.latent_observation_cache.replica(detach=True)
            # every 'param_learning_steps', do param learning (backprop_belief=False)
            if step % param_learning_steps == 0:
                total_vfe = sum(
                    self.inference_step(
                        timestep=t,
                        P_states=P_states,
                        backprop_belief=False
                    ) for t in self.states_cache.keys()
                )
                P_states = P_states.replica(detach=True, freeze=True)
                self.latent_observation_cache = self.latent_observation_cache.replica(detach=True)
                print(f"[Step {step}] VFE = {total_vfe:.6f}") if print_statements else None

                self.optimizer_registry.do_step(
                    key='inference',
                    loss=total_vfe
                )

        return total_vfe
    
    def inference_step(
            self,
            timestep,
            P_states,
            backprop_belief=False
        ):
        """
        Step function to perform belief updating over cache of beliefs.

        Args:
            timestep: Selected timestep with which we perform inference for
            P_states: Dict of Ph_t 
            backprop_belief: Boolean on whether or not to call belief optimizer.
        """
        qs_t = self.states_cache.get(timestep)
        qs_tm1 = self.states_cache.get(timestep - 1) if self.states_cache.has(timestep - 1) else None  # imperative and breaks responsibility but very verbose and clear
        qs_tp1 = self.states_cache.get(timestep + 1) if self.states_cache.has(timestep + 1) else None  # especially considering this whole function is quite noisy
        ps_t = P_states.get(timestep)
        print('------------IN ATOMIC TIMESTEP------------', timestep)
        qh_t = qs_t.get('h')
        ph_t = ps_t.get('h')
        qz_t = qs_t.get('z')
        pz_t = ps_t.get('z')
        print('qz_t before?', qz_t.mean)
        print('qh_t before?', qh_t.mean)
        # self.loss_module.include(
        #     ('qh_t vs ph_t', self.loss_module.kl,
        #     qh_t, ph_t))
        self.loss_module.include(
            ('qz_t vs pz_t', self.loss_module.kl,
            qz_t, pz_t))

        # TODO generalize this to different prior/would-be prior calculating functions
        # -------- prior from t-1 -------- 
        if qs_tm1 is not None:
            ps_t_from_tm1 = self.transition_model(qs_tm1)
            pz_t_from_tm1, ph_t_from_tm1 = ps_t_from_tm1.get('z'), ps_t_from_tm1.get('h')
            # TODO: FIX THIS VERBOSITY
            self.loss_module.include(
                ('qz_t vs tm1 -> pz_t', self.loss_module.kl,
                qz_t, pz_t_from_tm1))

        # -------- would-be h-prior at t+1 under current belief -------- 
        if qs_tp1 is not None:
            # print('qs_tp1.mean?', qs_tp1.get('h').mean)
            ps_at_tp1 = self.transition_model(qs_t)
            pz_at_tp1, ph_at_tp1 = ps_at_tp1.get('z'), ps_at_tp1.get('h')
            qh_tp1 = qs_tp1.get('h')
            qz_tp1 = qs_tp1.get('z')
            # TODO: FIX THIS VERBOSITY
            self.loss_module.include(
                ('qz_tp1 vs t -> pz_tp1', self.loss_module.kl,  # TODO: better naming convention here??
                qz_tp1, pz_at_tp1),
            )
        states_energy = self.loss_module.compute()

        # message from obs
        qz_t = qs_t.get('z')
        lat_o_pred = self.z_ae.decode(qz_t)
        lat_o_true = self.latent_observation_cache.get(timestep)
        self.loss_module.include(
            ('decode loss', self.loss_module.log_likelihood,
             lat_o_true, lat_o_pred.mean),  # TODO bad practice to .mean here. NOTE: LEARN VARIANCE BOOKMARK HERE
        )
        obs_negative_log = self.loss_module.compute()
        
        total_energy = states_energy - obs_negative_log

        if backprop_belief:
            self.optimizer_registry.do_step(
                key='states_cache',
                loss=total_energy
            )
        print('qz_t after?', qz_t.mean)
        print('qh_t after?', qh_t.mean)
        return total_energy

    def ground_wm(self):
        """
        Updates predictive WM to bootstrap it to transition model.
        for information as to whats really going on, see the update notes.
        """
        # format x
        print('-----------------GROUNDING WM STEP-----------------')
        pred_z = []
        a_beliefs = self.states_cache \
            .vmap(lambda s: s.get('a')) \
            .values()
        z_beliefs = self.states_cache \
            .vmap(lambda s: s.get('z')) \
            .values()
        h_beliefs = self.states_cache \
            .vmap(lambda s: s.get('h')) \
            .values()
        
        z = z_beliefs[0]
        h = h_beliefs[0]
        a = self.empty_action_init()
        for _ in z_beliefs:
            h, z, a = self.grounding_wm.forward_hza(h, z, a)
            pred_z.append(z)

        pred_z = type(pred_z[0]).concat(pred_z, 1) \
            .map(lambda x:
                x[:, :-self.atomic_timestep])  # exclude the final predicted z, as we dont have a belief for z at t+1
        z_beliefs = type(z_beliefs[0]).concat(z_beliefs, 1) \
            .map(lambda x:
                x[:, self.atomic_timestep:,]) # exclude the first belief z, which we do not predict for
        
        pred_h = h.map(lambda x:
            x[:, self.atomic_timestep:,]) # exclude the first state, which should still be in this state history
        h_beliefs = type(h_beliefs[0]).concat(h_beliefs, 1)

        # correct wm to accurately predict our beliefs.
        self.loss_module.include(
            ('wm_grounding', self.loss_module.kl, 
            pred_h, h_beliefs),
        )
        self.loss_module.include(
            ('wm_grounding', self.loss_module.kl, 
            pred_z, z_beliefs),
        )

        grounding_loss = self.loss_module.compute()

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
        """
        # create planning priors
        # s = self.states_cache.get_latest(1) \
        #     .values()[0]
        # prior_s = [s]
        # for _ in range(self.planning_horizon):
        #     s = self.grounding_wm(s)
        #     prior_s.append(s)


        
        # print('prior_s?', [(s.mean('h'), s.mean('z'), s.mean('a')) for s in prior_s])
        
    
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
        print('INTERLEAVED?', interleaved)

        for t in removed_atomic_timesteps:
            self.states_cache.remove(t) if self.states_cache.has(t) else None
            self.latent_observation_cache.remove(t) if self.latent_observation_cache.has(t) else None

        for t in interleaved:
            self.observation_cache.remove(t) if self.observation_cache.has(t) else None


    def empty_action_init(self):
        """helper method to lazily create empty a dim"""
        return Normal(
            z_mean_shape=(self.batch_size, self.atomic_timestep, self.a_dim,), 
            z_log_std_shape=(self.batch_size, self.atomic_timestep, self.a_dim,)
        )


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