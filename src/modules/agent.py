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
from energy_calc import Loss
from lib.action_head import create_action_head
from lib.action_mapping import CameraHierarchicalMapping
from lib.actions import ActionTransformer
from mineclip import MineCLIP
from optim import OptimRegistry
from state import StateNode
from torch import nn
from worldmodel import WorldModel


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
        clip_config=None,
        num_policies=None,
        state_depth=1,  # state depth, basically num of recurrent layers
        history=3,
        h_dim=8,
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
        self.latent_image_dim = clip_config.get('image_feature_dim', 512)
        self.amortized_inf = True
        self.history = history  # this means when inference is run, NOT inclusive of latest obs, there are these many past tiemsteps
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # observations and states cache.
        self.states_cache = {}
        self.observation_cache = {}
        self.latent_observation_cache = {}
        self.action_cache = {}
        self.latent_action_cache = {}

        # dimensionality and tensor shapes
        self.state_depth = state_depth
        self.h_dim = h_dim
        self.z_dim = self.h_dim
        self.a_dim = self.h_dim
        self.timestep_size = 1  # TODO clarify its use. right now, timestep_size refers to the time dim
        # of the incoming x when feeding into recurrent blocks, keeping at 1 for now

        # other non-model modules
        self.loss = Loss()
        self.optimizer_registry = OptimRegistry()

        # encoder to encode incoming observation(s)
        self.obs_encoder = MineCLIP(**clip_config)
        self.obs_encoder.load_ckpt(path, strict=True)
        logging.info("Successfully loaded MineCLIP encoder.")

        # models
        # maps latent observation to state, for now only takes latent obs of image
        self.z_encoder = nn.Sequential(
            nn.Linear(self.latent_image_dim, self.z_dim),
            nn.LayerNorm(self.z_dim),
            nn.ReLU()
        )  # P(z|o)
        # maps states to latent observation
        self.z_decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.latent_image_dim),
            nn.LayerNorm(self.latent_image_dim),
            nn.ReLU()
        ) # P(o|z).
        self.transition_model = WorldModel(
            recurrence_type="transformer",
            attention_memory_size=2 * self.timestep_size,
            hidsize=self.h_dim,
            n_recurrence_layers=state_depth,
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
            n_recurrence_layers=state_depth,
            timesteps=self.timestep_size,
        )

        self.pi_head = create_action_head(
            n_camera_bins=11,
            latent_size=h_dim,
            mapper_class=CameraHierarchicalMapping,
            temperature=2.0
        )
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        self._dummy_first = torch.from_numpy(np.array((False,))).to(device).unsqueeze(1)

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
    
    def initialize_state_node(self, t, z):
        # TODO: generalize further for during planning?
        if t == 0:
            initial = self.transition_model.initial_state(batchsize=1)  # list[tuple[None, tuple[torch.Tensor]]]
            h_masks, h_states = StateNode.unpack_mask_states(initial)
            self.world_model.state_mask = list(h_masks)
            h_states = list(h_states)
            # print('h_states shape init trans model?', [[t.shape for t in thing] for thing in h_states])

        else:
            state = self.states_cache[t - 1]
            a, z = state.a, state.z_mean
            _, h_states = self.forward_transition_model(a, z, state=state)
            # print('h_states shape trans model?', [[t.shape for t in thing] for thing in h_states])

        self.states_cache[t] = StateNode(
            h_state=h_states,
            z_value=z,
            a_dim=self.a_dim,
        )

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
        for t, o in self.observation_cache.items():
            if t not in self.states_cache:
                lat_o = self.obs_encoder.forward_image_features(o) # (image) -> (clip latent size=512)
                z = self.z_encoder(lat_o)  # (all latent_obs) -> (z)
                self.initialize_state_node(t, z)
                self.latent_observation_cache[t] = lat_o

    def belief_optim_wrap(self):
        """
        Helper method to convert all states in state cache to parameters, wrapping around them with an optimizer
        """
        # seperate optimizer for over beliefs over states
        all_params = []
        for s in self.states_cache.values():
            all_params.extend(list(s.h_modules.parameters()))
            all_params.append(s.z_mean)
        kwargs = dict(lr=0.005)
        self.optimizer_registry.register_optim('states_cache', all_params, optim_name='SGD', **kwargs)

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
        self.inference()

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
                P_states = {k: s.clone(freeze=True) for k, s in self.states_cache.items()}

            # random timestep selection, to run inference on
            timestep = random.choice(list(self.states_cache))
            print('-----------Selected timestep---------------', timestep)
            self.inference_step(timestep, P_states, backprop_belief=True)  # lr scheduling comes later. test first
            P_states = {k: s.clone(detach=True, freeze=True) for k, s in P_states.items()}
            self.latent_observation_cache = {k: o.clone().detach() for k, o in self.latent_observation_cache.items()}

            # every 'update_rounds', do param learning (backprop_belief=False)
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
                    self.optimizer_registry.do_step(
                        key='inference',
                        loss=total_vfe
                    )

        if len(self.states_cache) != 1:
            z = [s.z_mean for t, s in self.states_cache.items() if t != max(self.states_cache)]
            num_timesteps = len(z)
            z_all = torch.concat(z, dim=0)
            embed_dim = z_all.shape[-1]
            batch_size = int(z_all.shape[0] / num_timesteps)
            # TODO while we dont have actions we just dupe along embed dim
            a_z = torch.concat([z_all, z_all], dim=1)

            x = self.a_z_encoder(a_z).reshape(batch_size, num_timesteps, embed_dim)

            first_s = self.states_cache[min(self.states_cache)]
            print('min(self.states_cache)', min(self.states_cache), self.states_cache.keys())
            first_h =  first_s.get_h_states()
            print('first_h', first_h)
            init_h = self.world_model.initial_state(batchsize=1) # require init from self.wm cuz h from it must have certain num of timesteps
            init_mask, h_states = StateNode.unpack_mask_states(init_h)
            print('h states before', h_states)
            for idx, depth in enumerate(first_h):
                for jdx, thing in enumerate(depth):
                    print('thing shape', thing.shape, thing)
                    print('h_states[idx][jdx][-1] shape', h_states[idx][jdx][-1].shape)
                    h_states[idx][jdx][-1] = thing

            print('h states after', h_states)
            print('x shape', x.shape)
            print('xf_state', [[s.shape for s in something] for something in h_states])
            # TODO isnt the pruning before feeding into x handled by something else? either ways do it here
            most_recent_h = self.wm_attention_size - x.shape[1]
            h_states = [[h[:, -most_recent_h:, ...] for h in depth] for depth in h_states]
            print('h states after pruning', h_states)
            print('xf_states after pruning', [[s.shape for s in something] for something in h_states])
            _, state_mask, pred_h = self.world_model(x,
                state_mask=init_mask,
                xf_state=h_states,
                context={'first': self._dummy_first}
            )
            print('after step state_mask wm', state_mask)
            # self.world_model.state_mask = state_mask
            pred_h = StateNode.flatten(pred_h)
            prior_h = StateNode.flatten(h_states)

            total_loss = self.loss.kl_divergence(
                prior_h, pred_h
            ).mean()
            print('total wm loss?', total_loss)
            self.optimizer_registry.do_step(
                key='wm',
                loss=total_loss)
            print('backpropped wm!')

        print("Reached max inference steps.")

        return total_vfe

    

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
            _, ph_from_tm1 = self.forward_transition_model(a_tm1, qz_tm1, h=qh_tm1)
            ph_from_tm1 = StateNode.flatten(ph_from_tm1[0])  # TODO hardcoded to depth = 1 for now i think

        # -------- would-be h-prior at t+1 under current belief -------- 
        ph_at_tp1, qh_tp1 = None, None
        qh_t, qz_t, a_t = qs_t.get_h_states(), qs_t.z_mean, qs_t.a
        if s_tp1 is not None:
            _, ph_at_tp1 = self.forward_transition_model(a_t, qz_t, h=qh_t)
            ph_at_tp1 = StateNode.flatten(ph_at_tp1[0])
            qh_tp1 = s_tp1.flattened_h_states

        print('qh_t before??', qs_t.flattened_h_states)
        print('ph_t before??', ps_t.flattened_h_states)
        print('ph_at_tp1 before???', ph_at_tp1)
        print('qh_tp1 before???', qh_tp1)
        print('ph_from_tm1 before???', ph_from_tm1)
        print('qh_tm1 before???', qh_tm1)
        print('qz_t before???', qz_t)
        total_energy = self.loss.compute_vfe_node(
            qh_t=qs_t.flattened_h_states,
            ph_t=ps_t.flattened_h_states,
            ph_at_tp1=ph_at_tp1,
            qh_tp1=qh_tp1,
            ph_from_tm1=ph_from_tm1,
            qh_tm1=s_tm1.flattened_h_states if s_tm1 is not None else None,
            lat_o_pred=lat_o_pred,
            lat_o_true=lat_o_true,
        )
        print('total energy', total_energy)

        if backprop_belief:
            print('verify etc before backward step', [(mod.key.grad, mod.value.grad) for mod in qs_t.h_modules])
            self.optimizer_registry.do_step(
                key='states_cache',
                loss=total_energy
            )
            qs_t = self.states_cache[timestep]
            ps_t = P_states[timestep]
            print('qh_t after??', qs_t.flattened_h_states)
            print('ph_t after??', ps_t.flattened_h_states)
            print('ph_at_tp1 after???', ph_at_tp1)
            print('qh_tp1 after???', qh_tp1)
            print('ph_from_tm1 after???', ph_from_tm1)
            print('qh_tm1 after???', qh_tm1)
            print('qz_t after???', qz_t)

        return total_energy
    
    def prune(self):
        """
        Helper func to prune from caches if not in window.
        """
        window = range(self.curr_timestep - self.history, self.curr_timestep)
        print('window post-pruning', list(window))
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
    # print('return_dict', return_dict['obs'].keys())
    # print('return dict obs life stats', return_dict['obs']['life_stats'])
    # print('return dict obs use_item', return_dict['obs']['use_item'])
    pic = np.asarray(return_dict['obs']['pov']).astype(np.uint8)

    arr = cv2.resize(pic, (256, 160), interpolation=cv2.INTER_LINEAR)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    images = arr[np.newaxis, :, :, :]
    images = torch.from_numpy(images).to(device)
    images = images.permute(0, 3, 1, 2)
    print('images shape', images.shape, images.device)

    agent.forward(observations=images)