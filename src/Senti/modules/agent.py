"""
This module is meant to encapsulate the entire agent. 
Its scope contains all its components.
Its methods controls the flow of everything.
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

from Senti.modules.dataclasses.pomdpstate import POMDPState
from Senti.modules.optim.loss import EnergyAggregator
from Senti.modules.optim.optimregistry import OptimRegistry
from Senti.modules.utils.state import StateNode
from Senti.modules.worldmodel.worldmodel import WorldModel
from Senti.registry import AUTOENCODERS
from Senti.src.Senti.modules.utils.caches import Cache, StateTimestepCache, TensorCache


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
        agent_config = config.agent
        self.history = agent_config.history  # this means when inference is run, NOT inclusive of latest obs, there are these many past tiemsteps
        self.device = torch.device(agent_config.device)
        self.batch_size = agent_config.batch
        
        # observations and states cache.
        self.states_cache = StateTimestepCache(state_type=POMDPState)
        self.observation_cache = Cache(value_type=dict)
        self.latent_observation_cache = TensorCache()
        self.action_cache = TensorCache()
        self.latent_action_cache = TensorCache()

        # dimensionality and tensor shapes
        self.state_depth = agent_config.state_depth
        self.h_dim = agent_config.h_dim
        self.z_dim = agent_config.z_dim
        self.a_dim = self.h_dim
        self.timestep_size = 1  # TODO clarify its use. right now, timestep_size refers to the time dim
        # of the incoming x when feeding into recurrent blocks, keeping at 1 for now

        # other non-model modules
        self.loss = EnergyAggregator()
        self.optimizer_registry = OptimRegistry()

        # encoder to encode incoming observation(s)
        self.obs_autoencoder = AUTOENCODERS.build("NmmoAE", config)
        logging.info("Successfully loaded NmmoAE")

        # models
        self.latent_obs_dim = self.obs_autoencoder.hidden_size
        # maps latent observation to state, for now only takes latent obs of image
        self.z_encoder = nn.Sequential(
            nn.Linear(self.latent_obs_dim, self.z_dim),
            nn.LayerNorm(self.z_dim),
            nn.ReLU()
        )  # P(z|o)
        # maps states to latent observation
        self.z_decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.latent_obs_dim),
            nn.LayerNorm(self.latent_obs_dim),
            nn.ReLU()
        ) # P(o|z).
        self.transition_model = WorldModel(
            recurrence_type="transformer",
            attention_memory_size=2 * self.timestep_size,  # TODO man fuck me for using magic numbers
            hidsize=self.h_dim,
            n_recurrence_layers=self.state_depth,
            timesteps=self.timestep_size,
        )
        # for now, concat the latent prev action and latent obs together, then do preprocessing to map it to the 
        self.a_z_encoder = nn.Sequential(
            nn.Linear(self.z_dim + self.a_dim, self.h_dim),
            nn.LayerNorm(self.h_dim),
            nn.ReLU()
        )
        self.wm_attention_size = self.history + self.timestep_size
        self.world_model = WorldModel(
            recurrence_type="transformer",
            attention_memory_size=self.wm_attention_size,
            hidsize=self.h_dim,
            n_recurrence_layers=self.state_depth,
            timesteps=self.timestep_size,
        )

        self._dummy_first = torch.from_numpy(
            np.full((self.batch_size, 1), False, dtype=bool)).to(self.device)

        # optimization
        inference_params = (
            list(self.z_encoder.parameters()) +
            list(self.z_decoder.parameters()) +
            list(self.transition_model.parameters())
            # TODO: decide what to do with wm parameters
        )
        kwargs = dict(lr=0.005)
        self.optimizer_registry.register_optim(
            'inference', inference_params, **kwargs
        )

        wm_params = list(self.world_model.parameters())
        kwargs = dict(lr=0.005)
        self.optimizer_registry.register_optim(
            'wm', wm_params, **kwargs
        )

        self.curr_timestep = 0
        self.window = range(self.curr_timestep - self.history, self.curr_timestep)
        self.to(self.device)

    def initialize_state_node(self, t, z):
        if t == 0:
            h_masks, h_states = self.transition_model.initial_state(
                batch_size=self.batch_size)  # list[tuple[None, tuple[torch.Tensor]]]
            self.world_model.state_mask = list(h_masks)
            # h_states = list(h_states)
            
        else:
            state = self.states_cache.get(t - 1)
            am1, zm1 = state.a, state.z_mean
            _, h_states = self.forward_transition_model(am1, zm1, state=state)

        self.states_cache.add(t, StateNode(
            h_state=h_states,
            z_value=z,
            a_dim=self.a_dim,
        ))

    def forward_transition_model(self, a, z, state=None, h=None):
        """
        Helper function to run forward method of transition model.
        Can either take in the h -> list of tuples directly, or pass in StateNode object
        """
        assert state is not None or h is not None, 'Assertion failed, must pass in either h directly, or the StateNode object which contains it.'
        if state is not None:
            h = state.get_h_states()
        state_masks = [None] * self.state_depth
        a_z = torch.concat([z, z], dim=1)  # TODO wait till planning for us to predict actions
        x = self.a_z_encoder(a_z).unsqueeze(1)
        (pi_latent, vf_latent), _, h = self.transition_model(
            x,
            state_masks,
            h,
            context={'first': self._dummy_first}
        )

        return (pi_latent, vf_latent), h


    def update_states_cache(self):
        """
        Simple helper method to update all timesteps in observation cache.
        """
        for t in self.observation_cache.keys():
            if not self.latent_observation_cache.has(t):
                o = self.observation_cache.get(t)
                lat_o, others = self.obs_autoencoder.encode(o) # (image)
                z = self.z_encoder(lat_o)  # (all latent_obs) -> (z)
                self.initialize_state_node(t, z)
                self.latent_observation_cache.add(t, lat_o)

    def belief_optim_wrap(self):
        """
        Helper method to convert all states in state cache to parameters, wrapping around them with an optimizer
        """
        # seperate optimizer for over beliefs over states
        all_params = []
        for state_node in self.states_cache.values():
            all_params.extend(state_node.get_params())
        kwargs = dict(lr=0.5)
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
        # 1 - 4:
        t = self.curr_timestep
        self.observation_cache.add(t, observations)  # cache this observation
        self.inference()

        if t > 0:
            self.ground_wm()

        # 9. Act
        # self.pi_head.predict('output_from_transmodel')

        # cleanup: shift window and remove all keys not in the window
        self.curr_timestep += 1
        self.prune()

    def inference(
        self,
        max_update_steps: int = 100,
        update_rounds: int = 10,
        print_statements: bool = True,
    ) -> float:
        """
        Run the inference procedure for belief updates and model parameter learning.
        Currently in the form of amortized inference, perhaps when I have time will fit in non-amortized (param learning
        only when VFE converges.)

        Args:
            max_update_steps (int): Maximum number of message passing steps to perform.
            update_rounds (int): Number of belief updates before checking VFE and (possibly) updating model parameters.
        Returns:
            float: Final VFE after inference and (optional) learning.
        """

        for step in range(max_update_steps):
            if step == 0:  # first step, percieve for latest prior and wrap in optim
                self.update_states_cache()
                self.belief_optim_wrap()
                P_states = self.states_cache.replica(detach=True, freeze=True)

            # random timestep selection, to run inference on
            timestep = random.choice(list(self.states_cache.keys()))
            print('-----------Selected timestep---------------', timestep)
            self.inference_step(timestep, P_states, backprop_belief=True)  # lr scheduling comes later. test first
            P_states = P_states.replica(detach=True, freeze=True)
            self.latent_observation_cache = self.latent_observation_cache.clone(detach=True)

            # every 'update_rounds', do param learning (backprop_belief=False)
            if step % update_rounds == 0:
                total_vfe = sum(
                    self.inference_step(
                        timestep=t,
                        P_states=P_states,
                        backprop_belief=False
                    ) for t in self.states_cache.keys()
                )
                P_states = P_states.replica(detach=True, freeze=True)
                self.latent_observation_cache = self.latent_observation_cache.clone(detach=True)
                print(f"[Step {step}] VFE = {total_vfe:.6f}") if print_statements else None

                self.optimizer_registry.do_step(
                    key='inference',
                    loss=total_vfe
                )

        return total_vfe

    def ground_wm(self):
        """
        Updates predictive WM to bootstrap it to transition model.
        """
        # format x
        z = [s.z_mean for t, s in self.states_cache.items() if t != self.curr_timestep]
        num_future_timesteps = len(z)
        B, D = z[0].shape
        z_all = torch.concat(z, dim=0) # list[(B, D)] -> (T * B, D) so we can pass into nn
        a_z = torch.concat([z_all, z_all], dim=1)  # TODO while we dont have actions we just dupe along embed dim
        x = self.a_z_encoder(a_z)
        x = x.reshape(B, num_future_timesteps, D)

        # populate h
        first_s = self.states_cache.get(max(min(self.window), 0))
        first_h =  first_s.get_h_states()
        init_h = self.world_model.initial_state(batch_size=1) # require init from self.wm cuz h from it must have certain num of timesteps
        init_mask, h_states = StateNode.unpack_mask_states(init_h)
        for idx, depth in enumerate(first_h):
            for jdx, thing in enumerate(depth):
                h_states[idx][jdx][-1] = thing

        # TODO isnt the pruning before feeding into x handled by something else? either ways do it here
        most_recent_h = self.wm_attention_size - num_future_timesteps
        h_states = [[h[:, -most_recent_h:, ...] for h in depth] for depth in h_states]

        _, state_mask, pred_h = self.world_model(x,
            state_mask=[None],
            xf_state=h_states,
            context={'first': self._dummy_first}
        )

        pred_h = StateNode.flatten(pred_h)
        prior_h = StateNode.flatten(h_states)

        total_loss = self.loss.kl_divergence(
            prior_h, pred_h
        ).mean()

        self.optimizer_registry.do_step(
            key='wm',
            loss=total_loss)


    def inference_step(
            self,
            timestep,
            P_states,
            backprop_belief=False
        ):
        """
        Step function to perform belief updating over cache of beliefs.
        TODO: currently naming is messy, comments are sparse. see if you can make it clear what does what
        For each step of belief updating, we isolate one node, calculate the corresponding messages
        fed to it, then update it. 
        TODO 2: HOLY FUCK THIS IS MESSY
        TODO 3: SO MUCH REPEATED CODE GOD

        Args:
            timestep: Selected timestep with which we perform inference for
            P_states: Dict of Ph_t 
            backprop_belief: Boolean on whether or not to call belief optimizer.
        """
        state_calculations = []
        qs_t = self.states_cache.get(timestep)
        s_tm1 = self.states_cache.get(timestep - 1) if self.states_cache.has(timestep - 1) else None  # imperative and breaks responsibility but very verbose and clear
        s_tp1 = self.states_cache.get(timestep + 1) if self.states_cache.has(timestep + 1) else None  # especially considering this whole function is quite noisy
        ps_t = P_states.get(timestep)

        qh_t = qs_t.flattened_h_states
        ph_t = ps_t.flattened_h_states
        state_calculations.append(
            ('qh_t vs ph_t', self.loss.kl_divergence,
            qh_t, ph_t))

        # -------- prior from t-1 -------- 
        ph_t_from_tm1, qh_tm1 = None, None
        if s_tm1 is not None:
            qh_tm1, qz_tm1, a_tm1 = s_tm1.get_h_states(), s_tm1.z_mean, s_tm1.a
            _, ph_t_from_tm1 = self.forward_transition_model(a_tm1, qz_tm1, h=qh_tm1)
            ph_t_from_tm1 = StateNode.flatten(ph_t_from_tm1[0])  # TODO hardcoded to depth = 1 for now i think
            print('ph_t_from_tm1.shape after statenode flatten', ph_t_from_tm1.shape)
            print('qh_t shape?', qh_t.shape)
            state_calculations.append(
                ('qh_t vs tm1 -> ph_t', self.loss.kl_divergence,
                qh_t, ph_t_from_tm1))

        # -------- would-be h-prior at t+1 under current belief -------- 
        ph_at_tp1, qh_tp1 = None, None
        qh_t, qz_t, a_t = qs_t.get_h_states(), qs_t.z_mean, qs_t.a
        if s_tp1 is not None:
            _, ph_at_tp1 = self.forward_transition_model(a_t, qz_t, h=qh_t)
            ph_at_tp1 = StateNode.flatten(ph_at_tp1[0])  # TODO hardcoded to depth = 1 for now i think
            qh_tp1 = s_tp1.flattened_h_states
            state_calculations.append(
                ('qh_tp1 vs t -> ph_tp1', self.loss.kl_divergence,  # TODO: better naming convention here??
                qh_tp1, ph_at_tp1),
            )
        states_energy = self.loss.compute_energy(state_calculations)

        # message from obs
        qz_t = qs_t.z_mean
        lat_o_pred = self.z_decoder(qz_t)
        lat_o_true = self.latent_observation_cache.get(timestep)
        obs_negative_log = self.loss.compute_energy([
            ('decode loss', self.loss.log_likelihood, lat_o_pred, lat_o_true),
        ])
        
        total_energy = states_energy - obs_negative_log

        if backprop_belief:
            self.optimizer_registry.do_step(
                key='states_cache',
                loss=total_energy
            )

        return total_energy
    
    def prune(self):
        """
        Helper func to prune from caches if not in window.
        """
        new_window = range(self.curr_timestep - self.history, self.curr_timestep)
        removed_timesteps = list(set(self.window) - set(new_window))
        self.window = new_window

        print('REMOVED TIMESTEPS', removed_timesteps)

        for t in removed_timesteps:
            self.states_cache.remove(t) if self.states_cache.has(t) else None
            self.latent_observation_cache.remove(t) if self.latent_observation_cache.has(t) else None
            self.observation_cache.remove(t) if self.observation_cache.has(t) else None

    def planning(
            self,
            o_target
        ):
        """
        Method to run planning.
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