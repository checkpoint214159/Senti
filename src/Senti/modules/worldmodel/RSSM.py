import torch
from einops import rearrange
from torch import nn

from Senti.modules.config.config import Config, ConfigDict
from Senti.modules.dataclass.categorical import DepthGroupedCategoricalState
from Senti.modules.dataclass.normal import Normal
from Senti.modules.dataclass.pomdpstate import POMDPState
from Senti.registry import WORLDMODELS


@WORLDMODELS.register_module()
class RSSM(nn.Module):
    def __init__(self,
            config: Config):
        """
        RSSM worldmodel that can take in an external conditioning when generating z.
        """
        super().__init__()

        self.z_dim = config.z_dim
        self.h_dim: list[int] = config.h_dim
        self.num_GRU_layers = config.num_GRU_layers
        self.flattened_gc = config.h_dim[0] * config.h_dim[1]  # flattening the gc dimension of h
        self.flattened_h = self.flattened_gc * self.num_GRU_layers


        self.a_dim = config.a_dim
        self.external_dim = config.external_dim
        self.hidden = config.hidden_dim
        self.device = config.device
        self.h_groups, self.h_classes = self.h_dim[-2], self.h_dim[-1]
        
        # opted for GRUCell since GRU HATES our poor batchedTensors
        self.recurrent_core = nn.ModuleList([
            nn.GRUCell(
                input_size=(self.z_dim + self.a_dim) if i == 0 else self.flattened_gc,
                hidden_size=self.flattened_gc
            ) for i in range(self.num_GRU_layers)
        ])
        self.recurrent_core.flatten_parameters = lambda: None

        self.z_given_h = nn.Sequential(
            nn.Linear(self.flattened_h, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 2 * self.z_dim) # 2x for mean and std
        ) 

        self.z_given_h_ext = nn.Sequential(
            nn.Linear(self.flattened_h + self.external_dim, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 2 * self.z_dim)
        )


    def initial_h(self, *dims) -> DepthGroupedCategoricalState:
        """
        todo: enable it to be learned? i dont see the point though
        """
        G, C = self.h_groups, self.h_classes
        dims = (self.num_GRU_layers, ) + dims + (G, C)
        # dims = dims + (G, C)
        # print('dims in init h?', dims)
        h_t = torch.zeros(dims, device=self.device)
        return DepthGroupedCategoricalState(h_t)
    

    def wrap_z(self, z_mean_logstd,
        is_posterior=None, as_parameter=False) -> Normal:
        """wraps z in our normal"""
        mean, log_std = torch.split(z_mean_logstd, self.z_dim, dim=-1)
        log_std = torch.clamp(log_std, min=-5, max=2)
        std = torch.exp(log_std)
        return Normal(
            z_mean=mean,
            z_log_std=std,
            is_posterior=is_posterior,
            as_parameter=as_parameter,
        )


    def gru_step(self, x, t, prev_h: torch.Tensor):
        """
        Takes in x with timestep dimension at -2. Slices it then runs recurrent layers.
        If you want to run across all t at dim -2, you need to embed gru_step in a loop.

        TODO update this to just use gru
        """
        xt = x[..., t, :] # [..., az_emb]
        prev_h # [L, ..., h_emb]

        new_h_layers = []
        layer_input = xt
        for i, cell in enumerate(self.recurrent_core):
            h_next = cell(layer_input, prev_h[i, ...].squeeze(-2))  # [..., Len=1, h_dim]. squeeze for prev_h remvoes time
            new_h_layers.append(h_next)
            layer_input = h_next
        
        seed = torch.stack(new_h_layers, dim=0).unsqueeze(1)  # [..., h_emb] -> [L, T=1, ..., h_emb]
        last_layer_output = h_next
        return last_layer_output, seed


    def forward(self, mode: str, *args, **kwargs):
        """routes to various things, because functional call tragically only knows the forward method"""
        if mode == 'full':
            return self.forward_full(*args, **kwargs)
        elif mode == 'h':
            return self.forward_h(*args, **kwargs)
        elif mode == 'z':
            return self.forward_z(*args, **kwargs)
        else:
            raise ValueError("Invalid mode passed to RSSM forward.")

    
    def forward_full(self, s_t: POMDPState, a_t:torch.Tensor): 
        last_layer_out, h_t = self.forward_h(s_t, a_t)
        z = self.forward_z(h_t)
        return last_layer_out, h_t, z

    def forward_h(self, state:POMDPState, a:torch.Tensor) -> DepthGroupedCategoricalState:
        """
        h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        forwards one step only. GRUCell things. refuse to use GRU for longer sequences
        as 1. this doesnt run GRU only, there is a forward_z needed to generate next z anyway
        2. GRU doesnt play well with our vmap approach, i mean we can make it work but its a bit scuffed

        I know seeing a for-loop in a nn.Module is pretty cancer, but oh well surely this wont cause much of a slowdown
        
        notation:
        Len = sequence length
        D = 2 if bidirectional=True, 1 otherwise. Should always be 1 TODO lazy to account for this now
        N = num_layers.
        """
        G, C = self.h_groups, self.h_classes
        prev_z, prev_h = state.get('z'), state.get('h')
        z = prev_z.sample()

        x = torch.cat([z, a], dim=-1)  # [..., Len, az_emb]
        prev_h = prev_h.as_tensor(flatten_depth=False)  # [..., T, h_emb]. T SHOULD be 1 here.

        assert prev_h.shape[-2] == 1, 'Assertion failed. T should be 1 only, since it is the seed'

        last_layer_output, seed = self.gru_step(x, t=0, prev_h=prev_h)

        return last_layer_output, \
            DepthGroupedCategoricalState.from_flat_logits(seed, G, C)

    def forward_z(self, h_t: DepthGroupedCategoricalState,
            external:torch.Tensor | None = None) -> Normal:
        """
        (z_t | h_t) or (z_t | h_t, external_t)
        Still accept the whole POMDPState as an argument, to keep abstraction neat
        """
        is_posterior = False
        
        h_t = h_t.as_tensor() # -> [... G, C] into [..., G * C]
        if external is not None:
            is_posterior = True
            x = torch.cat([h_t, external], dim=-1)
            z = self.z_given_h_ext(x)
        else:
            z = self.z_given_h(h_t)
        
        return self.wrap_z(z, is_posterior)
    

if __name__ == "__main__":
    # brainrot test
    h_dim = [8, 8]
    flattened_h_dim = h_dim[0] * h_dim[1]
    z_dim = 8
    a_dim = 8
    obs_dim = 10
    state_depth = 3
    hidden_dim_internal = 20
    device = "cuda"
    test_config = ConfigDict(
        h_dim=h_dim,
        z_dim=z_dim,
        a_dim=a_dim,
        external_dim=obs_dim,
        device=device,
        hidden_dim=hidden_dim_internal,
        num_GRU_layers=state_depth
    )

    rssm = RSSM(test_config)
    example_h = DepthGroupedCategoricalState.from_flat_logits(
        flat_logits = torch.randn((state_depth, flattened_h_dim)),
        num_groups=h_dim[0],
        num_classes=h_dim[1],
    )
    example_obs = torch.randn((1, obs_dim))
    # print('example_h shape?', example_h.shape)
    # print('example_obs shape?', example_obs.shape)
    rssm.forward_z(
        h_t = example_h,
        external = example_obs,
    )

