import pytest
import torch
from mineclip import MineCLIP

from ActInfAgents.modules.worldmodel import WorldModel

# PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -v

# ---------------------------------------------
#         model fixtures and defaults 

# default configs for clip
resolution = [160, 256]
src_clip_config = {
    'arch': 'vit_base_p16_fz.v2.t2',
    'hidden_dim': 512,
    'image_feature_dim': 512,
    'mlp_adapter_spec': 'v0-2.t0',
    'pool_type': 'attn.d2.nh8.glusw',
    'resolution': resolution,
    'ckpt_path': '/mnt/e/ActInfAgents/weights/attn.pth'
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



def _make_mineclip(clip_config=src_clip_config):
    obs_encoder = MineCLIP(**clip_config)
    path = clip_config.get("ckpt_path", None)
    if path is not None:
        obs_encoder.load_ckpt(path, strict=True)
    return obs_encoder

@pytest.fixture
def mineclip_factory():
    """Returns a factory to build MineCLIP with custom configs."""
    return _make_mineclip

@pytest.fixture(scope="session")
def default_mineclip():
    """One shared MineCLIP instance for the whole test session."""
    return _make_mineclip()



def _make_transition_model(transition_config=src_transition_model):
    return WorldModel(**transition_config)

@pytest.fixture
def transition_model_factory():
    return _make_transition_model

@pytest.fixture(scope="session")
def default_transition_model():
    return _make_transition_model()

def _make_world_model(wm_config=src_wm_config):
    return WorldModel(**wm_config)

@pytest.fixture
def world_model_factory():
    return _make_world_model

@pytest.fixture(scope="session")
def default_world_model():
    return _make_world_model()

# ---------------------------------------------
#       optimizer, loss, etc. utilities



# ---------------------------------------------

