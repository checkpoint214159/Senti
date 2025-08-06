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
from gym3.types import DictType
from lib.action_head import make_action_head
from lib.action_mapping import CameraHierarchicalMapping
from lib.actions import ActionTransformer
from mineclip import MineCLIP
from state import StateNode
from torch import nn
from worldmodel import WorldModel

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

    TODO clean up agent responsibilities once we have moved to a baseline architecture we can play with
    """

    def __init__(self,
        clip_config=None,
        num_policies=None,
        state_depth=1,  # state depth, basically num of recurrent layers
        history=3,
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

        path = clip_config.pop('ckpt_path', None)
        image_dim = clip_config.get('image_feature_dim', 512)
        self.amortized_inf = True
        self.history = history  # this means when inference is run, NOT inclusive of latest obs, there are these many past tiemsteps
        self.state_depth = state_depth
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # observations and states cache.
        self.states_cache = {}

        self.observation_cache = {}
        self.latent_observation_cache = {}
        self.action_cache = {}
        self.latent_action_cache = {}

        # encoder to encode incoming observation(s)
        self.obs_encoder = MineCLIP(**clip_config)
        self.obs_encoder.load_ckpt(path, strict=True)
        logging.info("Successfully loaded MineCLIP encoder.")

        # maps latent observation to state, for now only takes latent obs of image
        self.z_encoder = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.LayerNorm(image_dim),
            nn.ReLU()
        )  # P(z|o)
        # maps states to latent observation
        self.z_decoder = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.LayerNorm(image_dim),
            nn.ReLU()
        ) # P(z|o).
        self.h_encoder = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.LayerNorm(image_dim),
            nn.ReLU()
        )  # P(h|z)
        self.h_decoder = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.LayerNorm(image_dim),
            nn.ReLU()
        )  # P(z|h)

        # # read the name
        # self.transition_model = TransitionModel(
        #     z_dim=512,
        #     a_dim=512,
        #     h_dim=512
        # ) # P(s_t+1 | s_t)

        # trying to use similar architecture for WM and Transition model, woop TODO just inject a TODO backhere
        self.transition_model = WorldModel(
            recurrence_type="transformer",
            attention_memory_size=2,
            hidsize=512,
            n_recurrence_layers=state_depth,
            timesteps=1,
        )
        # for now, concat the latent prev action and latent obs together, then do preprocessing to map it to the 
        # h dim: for now just 1024 -> 512
        self.a_z_encoder = nn.Sequential(
            nn.Linear(2 * image_dim, image_dim),
            nn.LayerNorm(image_dim),
            nn.ReLU()
        )

        self.world_model = WorldModel(
            recurrence_type="transformer",
            attention_memory_size=2,
            hidsize=512,
            n_recurrence_layers=state_depth,
            timesteps=1,
        )

        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        self.action_space = DictType(**action_space)
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        self.pi_head = make_action_head(
            self.action_space,
            self.transition_model.output_latent_size(),
            temperature=2.0
        )

        self._dummy_first = torch.from_numpy(np.array((False,))).to(device).unsqueeze(1)

        params = (
            list(self.z_encoder.parameters()) +
            list(self.z_decoder.parameters()) +
            list(self.transition_model.parameters())
            # TODO: decide what to do with wm parameters
        )
        self.inference_optimizer = torch.optim.AdamW(params, lr=0.005)

        params = list(self.world_model.parameters())
        self.wm_optimizer = torch.optim.AdamW(params, lr=0.005)
                
        self.num_policies = num_policies
        self.state_dicts = {}

        self.curr_timestep = 0

    @staticmethod
    def kl_divergence(mean1, mean2):
        """
        Helper func to calculate kl_divergence between two isotropic gaussians with variance 1, for now
        """
        return 0.5 * (-1 + 1 + (mean1 - mean2) ** 2)  # what even. idk bro, if gaussian is isotropic multivariate, its just this?

    @staticmethod
    def log_likelihood(o_true, o_pred):
        """
        Helper func to calculate log_likeihood of pred, w.r.t truth. Equivalent to above method for isotropic gaussians with
        var 1 too, see how we can change this
        """
        # assume we have uniform multivariate gaussian
        dist = torch.distributions.Normal(o_true, 1.0)
        return dist.log_prob(o_pred)
    
    @staticmethod
    def param_learning(optimizer, loss):
        """
        Simple helper method to update arbitrary optimizer (just so i dont need to repeat code)
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def update_states_cache(self):
        """
        Simple helper method to update all timesteps in observation cache.
        
        """
        for t, o in self.observation_cache.items():
            if t not in self.states_cache:
                lat_o = self.obs_encoder.forward_image_features(o)
                z = self.z_encoder(lat_o)
                if t == 0: # we have no prior hidden state
                    initial = self.world_model.initial_state(batchsize=1)  # list[tuple[None, tuple[torch.Tensor]]]
                    h_masks, h_states = StateNode.unpack_h(initial)
                    self.world_model.state_mask = list(h_masks)
                    h_states = list(h_states)

                else:  # let transition model predict
                    # TODO debug ts
                    state = self.states_cache[t - 1]
                    a, z = state.a, state.z_mean
                    h = list(zip([None] * self.state_depth, state.get_h_states()))
                    a_z = torch.concat([z, z], dim=1)  # TODO right now just dummy a
                    x = self.a_z_encoder(a_z).unsqueeze(1)
                    _, h = self.transition_model(
                        x, h, context={'first': self._dummy_first}
                    )
                    _, h_states = StateNode.unpack_h(h)

                self.states_cache[t] = StateNode(
                    h_state=h_states,
                    z_value=z,
                    a_dim=512,
                )
                self.latent_observation_cache[t] = lat_o

    def optim_wrap(self):
        """
        Helper method to convert all states in state cache to parameters, wrapping around them with an optimizer
        """
        # seperate optimizer over beliefs over states
        all_params = []
        for s in self.states_cache.values():
            all_params.extend(list(s.h_modules.parameters()))
            all_params.append(s.z_mean)
        self.beliefs_optimizer = torch.optim.SGD(all_params, lr=0.005)

    def forward(self,
                observations,
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
        self.observation_cache[t] = observations  # cache this observation
        if t > 0:
            self.inference(print_statements=False)
        else:
            self.inference()
        
        # 5 - 8:
        # self.planning()
        # converged = False
        # while not converged:
        #     for i in range(self.num_policies):
        #         pi = Q.sample  # some sample method
                
        #         # minimize vfe

        # # 8 - 10:
        # policy = Q.sample
        # action = policy.sample()

        # self.curr_timestep += 1

        # return action

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
                self.optim_wrap()
                P_states = {k: s.clone(freeze=True) for k, s in self.states_cache.items()}

            # random timestep selection, to run inference on
            timestep = random.choice(list(self.states_cache))
            # print('Selected timestep', timestep)
            self.inference_step(timestep, P_states, backprop_belief=True)  # lr scheduling comes later. test first
            P_states = {k: s.clone(detach=True, freeze=True) for k, s in P_states.items()}
            self.latent_observation_cache = {k: o.clone().detach() for k, o in self.latent_observation_cache.items()}

            # every 'update_rounds', do param learning (backprop_belief=True)
            if step % update_rounds == 0:
                total_vfe = sum(
                    self.inference_step(
                        timestep=t,
                        P_states=P_states,
                        backprop_belief=False
                    ) for t in self.states_cache
                )
                P_states = {k: s.clone(detach=True, freeze=True) for k, s in P_states.items()}
                self.latent_observation_cache = {k: o.clone().detach() for k, o in self.latent_observation_cache.items()}
                print(f"[Step {step}] VFE = {total_vfe:.6f}") if print_statements else None

                if self.amortized_inf:
                    # param learning for amortized case
                    self.param_learning(
                        optimizer=self.inference_optimizer,
                        loss=total_vfe
                    )

        # after all is said and done, wm learning based on beliefs. observes over whole state cache. stack along ficticious time dimension
        if len(self.states_cache) != 1:  # TODO better way to run this only if states cache has past timesteps
            z = self.states_cache[max(self.states_cache.keys())].z_mean
            z = z.unsqueeze(1)
            all_h = [[] for _ in range(self.state_depth)]
            # compile from all timesteps, into all_h
            for t, s in self.states_cache.items():
                if t != max(self.states_cache):
                    h = s.get_h_states()  # [(h_key, h_states), ...]
                    [all_h[depth].append(h[depth]) for depth in range(self.state_depth)]

            h_states = []
            for key_val_pairs in all_h:
                h_keys, h_values = zip(*key_val_pairs)  # unzip into two lists
                h_keys = torch.concat(h_keys, dim=1)     # shape: [..., t, ...]
                h_values = torch.concat(h_values, dim=1) # shape: [..., t, ...]
                h_states.append((h_keys, h_values))

            state_mask = self.world_model.state_mask
            h = list(zip(state_mask, h_states))
            _, pred_state_out = self.world_model(z, h, context={'first': self._dummy_first})
            _, pred_h = StateNode.unpack_h(pred_state_out)

            pred_h = StateNode.flatten(pred_h)
            h_states = StateNode.flatten(h_states)
            # loss against priors
            total_loss = self.kl_divergence(
                h_states, pred_h
            ).sum()
            self.param_learning(self.wm_optimizer, total_loss)

        print("Reached max inference steps.")

        return total_vfe

    def compute_vfe_node(
        self,
        qh_t: torch.Tensor,
        ph_t: torch.Tensor,
        lat_o_pred: torch.Tensor,
        lat_o_true: torch.Tensor,
        ph_from_tm1: torch.Tensor | None = None,
        qh_tm1: torch.Tensor | None = None,
        qh_tp1: torch.Tensor | None = None,
        ph_at_tp1: torch.Tensor | None = None,
        variance_prior: float = 1.0,
        variance_obs: float = 1.0
    ) -> torch.Tensor:
        """
        Computes the variational free energy (VFE) at a specific timestep `t` for a single state node.

        The VFE includes:
        - KL divergence between current belief `qh_t` and prior `ph_t`
        - KL divergence from message from the past (if ph_from_tm1 provided)
        - KL divergence from message from the future (if qh_tp1 and ph_at_tp1 provided)
        - Negative log-likelihood between predicted and true latent observations

        Args:
            qh_t (torch.Tensor): Current posterior mean (e.g., hidden state) at time `t`.
            ph_t (torch.Tensor): Prior mean at time `t` from dynamics.
            lat_o_pred (torch.Tensor): Predicted latent observation at time `t`.
            lat_o_true (torch.Tensor): Actual latent observation at time `t`.
            ph_from_tm1 (torch.Tensor, optional): Prior at `t` predicted from `t-1`.
            qh_tm1 (torch.Tensor, optional): Variational posterior at `t-1` (unused here, for symmetry).
            qh_tp1 (torch.Tensor, optional): Posterior at `t+1` (used for backward KL).
            ph_at_tp1 (torch.Tensor, optional): Prediction of `qh_tp1` from forward model.
            variance_prior (float): Variance for KL computations (assumes isotropic Gaussian).
            variance_obs (float): Variance used for log-likelihood (assumes isotropic Gaussian).

        Returns:
            torch.Tensor: Scalar tensor representing the total variational free energy at timestep `t`.
        """
        # Observation likelihood (negative log-likelihood)
        energy_obs = self.log_likelihood(lat_o_true, lat_o_pred)

        energy_states = 0.0

        # Message from previous timestep
        if ph_from_tm1 is not None:
            energy_states += self.kl_divergence(qh_t, ph_from_tm1)

        # Message from next timestep
        if qh_tp1 is not None and ph_at_tp1 is not None:
            energy_states += self.kl_divergence(qh_tp1, ph_at_tp1)

        # Current prior vs posterior at time t
        energy_states += self.kl_divergence(qh_t, ph_t)

        # Total variational free energy = state energy - observation energy
        total_energy = (energy_states - energy_obs).sum()
        return total_energy

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
        qs_t = self.states_cache[timestep]
        s_tm1 = self.states_cache.get(timestep - 1, None)
        s_tp1 = self.states_cache.get(timestep + 1, None)
        ps_t = P_states[timestep]
        
        # message from obs
        qz_t = qs_t.z_mean
        lat_o_pred = self.z_decoder(qz_t)
        lat_o_true = self.latent_observation_cache[timestep]

        # -------- prior from t-1 -------- 
        ph_from_tm1, qh_tm1 = None, None
        if s_tm1 is not None:
            qh_tm1, qz_tm1, a_tm1 = s_tm1.get_h_states(), s_tm1.z_mean, s_tm1.a
            # TODO for now dummy action
            a_z = torch.concat([qz_tm1, qz_tm1], dim=1)
            x = self.a_z_encoder(a_z).unsqueeze(1)
            h = list(zip([None] * self.state_depth, qh_tm1))
            _, h = self.transition_model(
                x, h, context={'first': self._dummy_first}
            )
            _, ph_from_tm1 = StateNode.unpack_h(h)  # annoying 1 size tuple if state depth one
            ph_from_tm1 = StateNode.flatten(ph_from_tm1[0])

        # -------- would-be h-prior at t+1 under current belief -------- 
        ph_at_tp1, qh_tp1 = None, None
        qh_t, qz_t, a_t = qs_t.get_h_states(), qs_t.z_mean, qs_t.a
        if s_tp1 is not None:
            # TODO for now dummy action
            a_z = torch.concat([qz_t, qz_t], dim=1)
            x = self.a_z_encoder(a_z).unsqueeze(1)
            h = list(zip([None] * self.state_depth, qh_t))
            _, h = self.transition_model(
                        x, h, context={'first': self._dummy_first}
                    )
            _, ph_at_tp1 = StateNode.unpack_h(h)
            ph_at_tp1 = StateNode.flatten(ph_at_tp1[0])
            qh_tp1 = s_tp1.flattened_h_states

        # for k, thing in dict(
        #     ph_t=ph_t, 
        #     qs_t=qs_t, 
        #     po_t=o_true,
        #     o_pred=o_pred,
        #     prev_ph_t=prev_ph_t,
        #     next_qh_t=next_qh_t,
        # ).items():
        #     if thing is not None:
        #         print(k, thing.min(), thing.max())

        total_energy = self.compute_vfe_node(
            qh_t=qs_t.flattened_h_states,
            ph_t=ps_t.flattened_h_states,
            ph_at_tp1=ph_at_tp1,
            qh_tp1=qh_tp1,
            ph_from_tm1=ph_from_tm1,
            qh_tm1=s_tm1.flattened_h_states if s_tm1 is not None else None,
            lat_o_pred=lat_o_pred,
            lat_o_true=lat_o_true,
        )

        if backprop_belief:
            qs_t = self.states_cache[timestep]
            self.param_learning(optimizer=self.beliefs_optimizer, loss=total_energy)

        return total_energy
    
    def prune(self):
        """
        Helper func to prune from caches if not in window.
        """
        window = range(self.curr_timestep - self.history, self.curr_timestep)
        print('window', list(window))
        self.states_cache = {
            t: v for t, v in self.states_cache.items() if t in window
        }
        self.latent_observation_cache = {
            t: v for t, v in self.latent_observation_cache.items() if t in window
        }
        self.observation_cache = {
            t: v for t, v in self.observation_cache.items() if t in window
        }
 
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


resolution = [160, 256]
clip_config = {
    'arch': 'vit_base_p16_fz.v2.t2',
    'hidden_dim': 512,
    'image_feature_dim': 512,
    'mlp_adapter_spec': 'v0-2.t0',
    'pool_type': 'attn.d2.nh8.glusw',
    'resolution': resolution,
    'ckpt_path': '/mnt/e/AutonoMC/weights/attn.pth'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(clip_config=clip_config).to(device)

noop_action = {
    "attack": [0],
    "back": [0],
    "forward": [0],
    "jump": [0],
    "left": [0],
    "right": [0],
    "sneak": [0],
    "sprint": [0],
    "use": [0],
    "drop": [0],
    "inventory": [0],
    "hotbar.1": [0],
    "hotbar.2": [0],
    "hotbar.3": [0],
    "hotbar.4": [0],
    "hotbar.5": [0],
    "hotbar.6": [0],
    "hotbar.7": [0],
    "hotbar.8": [0],
    "hotbar.9": [0],
    "camera": [[0.0, 0.0]]
}

for i in range(10):
    response = requests.post(
        'http://localhost:8000/take_step',
        json={
            'action': noop_action,
        }
    )

    return_dict = json.loads(response.content)
    print('return_dict', return_dict['obs'].keys())
    print('return dict obs life stats', return_dict['obs']['life_stats'])
    print('return dict obs use_item', return_dict['obs']['use_item'])
    pic = np.asarray(return_dict['obs']['pov']).astype(np.uint8)

    arr = cv2.resize(pic, (256, 160), interpolation=cv2.INTER_LINEAR)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    images = arr[np.newaxis, :, :, :]
    images = torch.from_numpy(images).to(device)
    images = images.permute(0, 3, 1, 2)
    print('images shape', images.shape, images.device)

    agent.forward(observations=images)