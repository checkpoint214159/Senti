from modules.worldmodel import WorldModel

import pytest
import torch
from mineclip import MineCLIP

# PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -v

# default configs for clip
resolution = [160, 256]
src_clip_config = {
    'arch': 'vit_base_p16_fz.v2.t2',
    'hidden_dim': 512,
    'image_feature_dim': 512,
    'mlp_adapter_spec': 'v0-2.t0',
    'pool_type': 'attn.d2.nh8.glusw',
    'resolution': resolution,
    'ckpt_path': '/mnt/e/AutonoMC/weights/attn.pth'
}
latent_image_dim = src_clip_config.get('image_feature_dim', 512)
timestep_size = 1
state_depth = 1
h_dim =latent_image_dim

# default configs for transformer
src_wm_config = dict(
    recurrence_type="transformer",
    attention_memory_size=2 * timestep_size,
    hidsize=h_dim,
    n_recurrence_layers=state_depth,
    timesteps=timestep_size,
)

# default configs for wm
history = 3
wm_attention_size = history + timestep_size
src_transition_model = dict(
    recurrence_type="transformer",
    attention_memory_size=wm_attention_size,
    hidsize=h_dim,
    n_recurrence_layers=state_depth,
    timesteps=timestep_size,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def make_mineclip():
    def _make_mineclip(clip_config=None):
        if clip_config is not None:
            src_clip_config = clip_config
        obs_encoder = MineCLIP(**src_clip_config)
        path = src_clip_config.pop('ckpt_path', None)
        obs_encoder.load_ckpt(path, strict=True)
        return obs_encoder
    return _make_mineclip

@pytest.fixture
def make_transition_model():
    def _make_transition_model(transition_config=None):
        if transition_config is not None:
            src_transition_model = transition_config
        transition_model = WorldModel(**src_transition_model)
        return transition_model
    return _make_transition_model

@pytest.fixture
def make_world_model():
    def _make_world_model(wm_config=None):
        if wm_config is not None:
            src_wm_config = wm_config
        wm = WorldModel(**src_wm_config)
        return wm
    return _make_world_model

