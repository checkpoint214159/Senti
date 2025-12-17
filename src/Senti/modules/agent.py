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

    For now, only amortized inference will be supported. For research purposes i will try and make non-amortized easily
    integrable with the overall flow. (not happening bud)
    """

    def __init__(self,
        config: DictConfig,
    ):
        """
        The agent contains the following modules:
            - CLIP encoder for o -> lat_o
            - state encoder, then decoder for lat_o -> z -> lat_o
            - TransitionModel (rename to dynamics soon?) h_tp1 = f(h_tm1, z_tm1, a_tm1)
            - 
        """
        # admin stuff
        super().__init__()
        self.config = config
        self.history = self.config.history  # this means when inference is run, NOT inclusive of latest obs, there are these many past tiemsteps
        self.device = torch.device(self.config.device)
        self.batch_size = self.config.batch_size
        self.inference_steps = self.config.inference_steps
        self.param_learning_steps = self.config.param_learning_steps
        nmmo_agent_check_config(self.config)
        
        # observations and states cache.
        self.states_cache = StateTimestepCache(value_type=POMDPState)
        self.observation_cache = Cache(value_type=dict)
        self.latent_observation_cache = StateTimestepCache(value_type=Normal)
        self.action_cache = TensorCache()
        self.latent_action_cache = TensorCache()

        # dimensionality and tensor shapes
        self.state_depth = self.config.state_depth
        self.h_dim = self.config.h_dim
        self.z_dim = self.config.z_dim
        self.a_dim = self.z_dim

        # other non-model modules
        self.loss_module = DistributionLosses()
        self.optimizer_registry = OptimRegistry()

        # encoder to encode incoming observation(s)
        self.obs_autoencoder = AUTOENCODERS.build("NmmoObsAE", config.NmmoObsAE)
        logging.info("Successfully loaded NmmoAE")
        self.z_ae = AUTOENCODERS.build("zAE", config.zAE)

        # maps latent observation to state, for now only takes latent obs of image
        self.transition_model = WORLDMODELS.build("nmmoWmAdapter", config.transition_model)
        logging.info("Successfully loaded transition_model")

        self.grounding_wm = WORLDMODELS.build("nmmoWmAdapter", config.grounding_wm)
        logging.info("Successfully loaded grounding_wm")

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

    def initialize_state_node(self, t: int, z: Normal):
        if t == 1:
            _, h_states = self.transition_model.initial_state()  # h_0
     
        else:
            # calculate next h using previous state
            # print('IN INT STATE NODE T > 1, RUNNING TRANSITION MODEL. T IS', t)
            state = self.states_cache.get(t - 1)
            # print('h before, z before', state.get('h').mean, state.get('z').mean)
            latent, h_states = self.transition_model(state)
            # print('h after', h_states.mean)


        # print('z and h states shape?', h_states.shape, z.shape)
        self.states_cache.add(t, POMDPState(
            h=h_states,
            z=z,
            a=self.empty_action_init()).to(self.device)
        )

    def update_latent_obs(self, timesteps:list[int], atomic_num:int):
        """
        TODO: assumes that the obs encoder operates on each timestep seperately, i.e it doesnt support us just
        stacking the tensor and doing one pass through. this should change
        """
        obs = []
        for t in timesteps:
            o = self.observation_cache.get(t)
            agent_embed_normal, others \
                    = self.obs_autoencoder.encode(o) # TODO: clarify what is the difference between the two.
            obs.append(agent_embed_normal)

        stacked = type(obs[0]).concat_timesteps(obs)
        self.latent_observation_cache.add(
                atomic_num, stacked.to(self.device))
        

    def update_states_cache(self):
        """
        Simple helper method to update all timesteps in observation cache.
        """
        for t in self.latent_observation_cache.keys():
            if not self.states_cache.has(t):
                e: Normal = self.latent_observation_cache.get(t)
                # print('e from obs encoder', e.mean)
                z: Normal = self.z_ae.encode(e)  # (all latent_obs) -> (z)
                # print('z from z encoder', z.mean)
                self.initialize_state_node(t, z)
                

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
                2. Use it to update recurrent beliefs h at timestep t
                3. From existing beliefs h in the belief cache, do belief updating
                4. Every n steps of belief updating, calculate VFE and backprop model parameters
                5. Backprop world model based on new sequence of past states post-vfe
            Planning:
                6. Use WM to generate prior over future states
                7. Sample a policy (params) for action head, then do rollouts and calculate EFE
                8. Use EFE to update belief over policies
                9. Repeat until convergence, then sample policy and predict next action
        Then increment timestep by 1.
        """
        # 1 - 5:
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
                
                if self.grounding_wm.valid_atomic_timestep_count(self.atomic_count):
                    self.ground_wm()

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
            # print("------------------POST INFERENCE RESULT:------------------")
            # print([v.get('h').mean for v in self.states_cache.values()])
            self.inference_step(timestep, P_states, backprop_belief=True)  # lr scheduling comes later. test first
            P_states = P_states.replica(detach=True, freeze=True)
            self.latent_observation_cache = self.latent_observation_cache.replica(detach=True)
            # print("------------------POST INFERENCE RESULT:------------------")
            # print([v.get('h').mean for v in self.states_cache.values()])
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

    def ground_wm(self):
        """
        Updates predictive WM to bootstrap it to transition model.
        for information as to whats really going on, see the update notes.
        """
        # format x
        print('-----------------GROUNDING WM STEP-----------------')
        all_z_state = self.states_cache.get_latest(self.grounding_wm.x_atomic) \
            .vmap(lambda s: s.get('z')) \
            .values()
        z = all_z_state[0].__class__.concat(all_z_state, 1)  # list of length T-1, concat (B, D) tensors to get (B, T, D) 

        first_h = self.states_cache.get(max(min(self.atomic_window), 1)).get('h')
        h_masks, h_states = self.grounding_wm.initial_state(first_h) # require init from self.wm cuz h from it must have certain num of timesteps
        init_s = POMDPState(
                h=h_states,
                z=z,
                a=self.empty_action_init(),
            ).to(self.device)

        latent, pred_h = self.grounding_wm(
            init_s
        )

        # self.loss_module.include(
        #         ('wm_grounding', self.loss_module.kl_divergence, 
        #         qh_tp1, pred_h),
        #     )
        # grounding_loss = self.loss_module.compute()

        # self.optimizer_registry.do_step(
        #     key='wm',
        #     loss=grounding_loss)


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
        # print('qh_t.mean?', qh_t.mean)
        # print('ph_t.mean?', ph_t.mean)
        self.loss_module.include(
            ('qh_t vs ph_t', self.loss_module.kl,
            qh_t, ph_t))
        self.loss_module.include(
            ('qz_t vs pz_t', self.loss_module.kl,
            qz_t, pz_t))

        # TODO generalize this to different prior/would-be prior calculating functions
        # -------- prior from t-1 -------- 
        if qs_tm1 is not None:
            # print('qh_tm1.mean?', qs_tm1.get('h').mean)
            pz_t_from_tm1, ph_t_from_tm1 = self.transition_model(qs_tm1)
            # print('ph_t_from_tm1.mean?', ph_t_from_tm1.mean)
            # print('latent z from ph_t_from_tm1?', latent.mean)
            # print('latent z from qs_t?', qs_t.get('z').mean)
            self.loss_module.include(
                ('qh_t vs tm1 -> ph_t', self.loss_module.kl,
                qh_t, ph_t_from_tm1))
            self.loss_module.include(
                ('qz_t vs tm1 -> pz_t', self.loss_module.kl,
                qz_t, pz_t_from_tm1))

        # TODO generalize this to different prior/would-be prior calculating functions
        # -------- would-be h-prior at t+1 under current belief -------- 
        if qs_tp1 is not None:
            # print('qs_tp1.mean?', qs_tp1.get('h').mean)
            pz_at_tp1, ph_at_tp1 = self.transition_model(qs_t)
            qh_tp1 = qs_tp1.get('h')
            qz_tp1 = qs_tp1.get('z')
            # print('ph_at_tp1.mean?', ph_at_tp1.mean)
            # print('latent z from ph_at_tp1?', latent.mean)
            # print('latent z from qs_tp1?', qs_tp1.get('z').mean)
            self.loss_module.include(
                ('qh_tp1 vs t -> ph_tp1', self.loss_module.kl,  # TODO: better naming convention here??
                qh_tp1, ph_at_tp1),
            )
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

        return total_energy
    
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

    def planning(
            self,
            o_target
        ):
        """
        Method to run planning. TODO destroy this?
        """
        total_efe = {}
        for i in range(max_policies_sampled):
            pi = policies[i].unsqueeze(0)  # shape: (1, policy_dim)
            ph_t = self.states_cache[self.curr_timestep]

            efe = 0
            for t in range(horizon):
                # Predict next state
                ps_next = self.transition_model(ph_t, pi)

                # Predict observation
                po_next = self.z_decoder(ps_next)
                o_dist = Normal(po_next, 1.0)
                o_sample = o_dist.rsample()

                # epistemic value
                qs_next = self.z_encoder(o_sample)
                epistemic = self.kl_divergence(qs_next, ps_next)

                # instrumental value
                energy_obs = self.log_likelihood(o_target, po_next)  # for now, very simple calc

                efe += epistemic - energy_obs
                ph_t = ps_next.detach()  # move to next state (prevent backprop across time)

            total_efe[i] = efe

        return total_efe  # shape: (N,)

    def compute_efe_node(self):
        """
        Compute EFE. See what to do first
        """

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