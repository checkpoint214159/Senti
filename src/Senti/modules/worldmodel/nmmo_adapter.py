import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from torch import nn

from Senti.modules.dataclasses.normal import DepthNormal
from Senti.modules.dataclasses.pomdpstate import POMDPState
from Senti.modules.dataclasses.recurrentkvstate import RecurrentKVState
from Senti.modules.worldmodel.worldmodel import WorldModel
from Senti.registry import WORLDMODELS


@WORLDMODELS.register_module()
class nmmoWmAdapter(WorldModel):
    """
    hi this is me ben
    im not gonna touch WM's internals at all
    however i will force this adapter to internally create an return our goated dataclasses
    because i am tired of dealing with his nested list tuple thingamajig

    this still respects its module's boundaries: i am not making e.g ResidualRecurrentBlocks
    adhere to my custom typing. basically any module that does the actual computation is sacred from our typing
    
    """
    def __init__(self, wm_config: DictConfig):
        super().__init__(**wm_config)
        self.attention_memory_size = wm_config.attention_memory_size  # added this line for easier access
        self.timesteps = wm_config.timesteps  # added this line for easier access
        self.batch_size = wm_config.batch_size
        self.device = wm_config.device
        latent_az_encoder_config =  getattr(wm_config, "latent_az_encoder", None)
        
        self.a_z_encoder = nn.Sequential(
            nn.Linear(latent_az_encoder_config.az_dim, self.hidsize),
            nn.LayerNorm(self.hidsize),
            nn.ReLU()
        )
    
        self._dummy_first = torch.from_numpy(
            np.full((self.batch_size, self.timesteps), False, dtype=bool)).to(self.device)


    def forward(self, state: POMDPState) \
            -> tuple[torch.Tensor, torch.Tensor, DepthNormal[RecurrentKVState]]:
        """
        Helper function to run forward method of WM.
        Basically adapts the args into a digestible form for WM without changing its internals
        """
        am, zm = state.mean('a'), state.mean('z')
        state_masks = [None] * self.depth
        a_z = torch.concat([zm, zm], dim=-1)  # TODO wait till planning for us to predict actions
        x = self.a_z_encoder(a_z)
        h = self.resolve_missing_timesteps(x, state.get('h'))
        h_raw = h.raw()
        next_x, state_masks, h_raw = super().forward(
            x,
            state_masks,
            h_raw,
            context={'first': self._dummy_first}
        )

        h = state.get('h')
        return next_x, DepthNormal(
            depth=h.depth,
            values=[RecurrentKVState(*kv) for kv in h_raw],
            stateclass=h.stateclass,
        )
    
    def resolve_missing_timesteps(self, x:torch.Tensor, h: DepthNormal):
        """
        to prevent erroring out, if the sum of timesteps provided by x and h are not equal to or greater than
        this WM's timesteps', pad h by repeating its last value over and over. the only thing this is crucial
        for, is for bias creation later in the wm. dont worry about understanding this, just know that there is
        noqa if you touch this function.
        """
        x_timesteps = x.shape[1] / self.timesteps
        h_timesteps = h.shape[1] / self.timesteps
        atomic_atm = self.attention_memory_size / self.timesteps
        assert x_timesteps.is_integer()
        assert h_timesteps.is_integer()
        assert atomic_atm.is_integer()
        x_timesteps, h_timesteps, atomic_atm = int(x_timesteps), int(h_timesteps), int(atomic_atm)
        print("x_timesteps, h_timesteps, atomic_atm", x_timesteps, h_timesteps, atomic_atm)
        if x_timesteps + h_timesteps < atomic_atm:
            newshape = [1] * h.dim
            newshape[1] = atomic_atm - x_timesteps
            h = h.repeat(newshape)
        return h

    
    def initial_state(self, first: DepthNormal|None = None):
        """
        helper func to generate initial state.
        additional args is added, so that a singular value can be passed in, to populate the otherwise meaningless state
        this is because the user might have an idea of what the first state should be, but not require editing it themselves
        to suit the timestep needs of this class.
        """
        h_masks, raw_h = super().initial_state(self.batch_size)
        if first is not None:
            # hardcode assume that dimmension = 1 is where timestep dim is. bad practice. whoop!
            shape_like = [1] * first.dim
            shape_like[1] = self.timesteps
            h_states = first.repeat(shape_like)
        else:
            h_states = DepthNormal(
                depth=len(raw_h),
                values=[RecurrentKVState(h[0], h[1], init_log_std=0.5)
                    for h in raw_h],
                stateclass=RecurrentKVState
            )
        return h_masks, h_states
