import numpy as np
import torch
from omegaconf.dictconfig import DictConfig
from torch import nn

from Senti.modules.dataclasses.normal import DepthNormal, Normal
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
    
    The core of this module is its forward, which acts on either the POMDP state dataclass, which has h, z, and a baked into it,
    or you can call forward_hza with it seperately and leave the creation of state to us
    Note we do not care how a and z are created, that is up to implementors, we just concatenate them together and feed it into az_encoder
    to map it to h-dim
    """
    def __init__(self, wm_config: DictConfig):
        super().__init__(**wm_config)
        self.cache_keep_len = wm_config.cache_keep_len  # added this line for easier access
        self.atomic_size = wm_config.atomic_size
        assert (self.cache_keep_len / self.atomic_size).is_integer()
        self.batch_size = wm_config.batch_size
        self.device = wm_config.device
        self.az_dim = wm_config.az_dim
        
        # TODO: make this into an autoencoder. you lazy lazy man
        self.a_z_encoder = nn.Sequential(
            nn.Linear(self.az_dim, self.hidsize),
            nn.LayerNorm(self.hidsize),
            nn.ReLU()
        )

        self.a_z_decoder = nn.Sequential(
            nn.Linear(self.hidsize, self.az_dim),
            nn.LayerNorm(self.az_dim),
            nn.ReLU()
        )
    
    
    def generate_dummy_first(self, timesteps):
        return torch.from_numpy(
            np.full((self.batch_size, timesteps), False, dtype=bool)).to(self.device)

    def forward(self, state: POMDPState) \
            -> tuple[torch.Tensor, torch.Tensor, DepthNormal[RecurrentKVState]]:
        """
        Helper function to run forward method of WM.
        Basically adapts the args into a digestible form for WM without changing its internals
        This forward is called with the dataclass POMDPState, so it will return dataclass POMDPState.
        If you wish to call it with h, z and a
        """
        am, zm = state.mean('a'), state.mean('z')
        a_t = am.shape[2]
        z_t = zm.shape[2]
        assert a_t + z_t == self.az_dim, "Assertion failed, timesteps from a and z dont sum to self.az_dim"
        state_masks = [None] * self.depth
        a_z = torch.concat([zm, zm], dim=-1)  # TODO wait till planning for us to predict actions
        
        x = self.a_z_encoder(a_z)

        h_raw = state.get('h').raw()
        next_x, state_masks, h_raw = super().forward(
            x,
            state_masks,
            h_raw,
            context={'first': self.generate_dummy_first(x.shape[1])}
        )
        # from comment above: then we can disassemble az properly too.
        next_az = self.a_z_decoder(next_x)
        next_a, next_z = next_az[:, :, :a_t], next_az[:, :, a_t:]

        h = state.get('h')
        # until planning is set up, dont return action
        s = POMDPState(
            h=DepthNormal(
                depth=h.depth,
                values=[RecurrentKVState(*kv) for kv in h_raw],
                stateclass=h.stateclass,
            ).to(self.device),
            z=Normal.from_value(next_z).to(self.device),
            a=Normal.from_value(next_a).to(self.device),
        )
        return s
    
    def forward_hza(self, h: DepthNormal, z: Normal, a: Normal):
        """
        Helper func to run forward, whilst passing and recieving the arguments seperately
        might use this when creating the state is an additional step, as you may be sourcing everything from
        alll sorts of places
        """
        pomdp = POMDPState(h=h,z=z,a=a)
        pomdp = self.forward(pomdp)
        return pomdp.get('h'), pomdp.get('z'), pomdp.get('a')
    
    @property
    def x_atomic(self):
        return int(self.cache_keep_len / self.atomic_size)

    # def valid_x(self, x:torch.Tensor):
    #     "correct way of interfacing without extracting attribute of nmmoWmAdapter."
    #     assert isinstance(x, torch.Tensor)
    #     return x.shape[1] == self.x_timesteps
        
    # def valid_timestep_count(self, raw_timestep:int):
    #     "variant of valid_x for just the raw timestep count."
    #     return raw_timestep >= self.x_timesteps
    
    # def valid_atomic_timestep_count(self, atomic_timestep:int):
    #     "variant of valid_x for just the atomic_timestep count."
    #     return atomic_timestep >= self.x_atomic


    # def resolve_missing_timesteps(self, x:torch.Tensor, h: DepthNormal):
    #     """
    #     DEPRECIATED: incorrect way of resolving a different issue. see if we delete it soon.

    #     to prevent erroring out, if the sum of timesteps provided by x and h are not equal to or greater than
    #     this WM's timesteps', pad h by repeating its last value over and over. the only thing this is crucial
    #     for, is for bias creation later in the wm. dont worry about understanding this, just know that there is
    #     noqa if you touch this function.
    #     """
    #     x_timesteps = x.shape[1] / self.atomic_size
    #     h_timesteps = h.shape[1] / self.atomic_size
    #     atomic_atm = self.cache_keep_len / self.atomic_size
    #     assert x_timesteps.is_integer()
    #     assert h_timesteps.is_integer()
    #     assert atomic_atm.is_integer()
    #     x_timesteps, h_timesteps, atomic_atm = int(x_timesteps), int(h_timesteps), int(atomic_atm)
    #     # print("x_timesteps, h_timesteps, atomic_atm", x_timesteps, h_timesteps, atomic_atm)
    #     if x_timesteps + h_timesteps < atomic_atm:
    #         newshape = [1] * h.dim
    #         newshape[1] = atomic_atm - x_timesteps
    #         h = h.repeat(newshape)
    #     return h

    
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
            shape_like[1] = self.x_atomic
            h_states = first.repeat(shape_like)
        else:
            h_states = DepthNormal(
                depth=len(raw_h),
                values=[RecurrentKVState(h[0], h[1], init_log_std=0.5)
                    for h in raw_h],
                stateclass=RecurrentKVState
            )
        return h_masks, h_states
