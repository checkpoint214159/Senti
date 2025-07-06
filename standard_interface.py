import requests
import json
import numpy as np
import cv2
import torch
from mineclip import MineCLIP
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra

from tqdm import tqdm
from mineclip.mineagent import features as F
from mineclip import SimpleFeatureFusion, MineAgent, MultiCategoricalActor
from mineclip.mineagent.batch import Batch
from mineclip import CombatSpiderDenseRewardEnv

from scipy.stats import multivariate_normal


# use this script to test out pinging the mc instance server
resolution = [160, 256]
response = requests.post(
    'http://localhost:8000/take_step',
    json={
        'action': [0, 0, 0, 12, 12, 0, 0, 0],
    }
)


return_dict = json.loads(response.content)

pic = np.asarray(return_dict['obs']['rgb']).astype(np.uint8)
print('pic shape', pic.shape)
pic = pic.transpose((1, 2, 0)) # (C, H, W) -> (H, W, C)

arr = np.asarray(pic).astype(np.uint8)
arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_obs(env_obs):
    """
    Here you preprocess the raw env obs to pass to the agent.
    Preprocessing includes, for example, use MineCLIP to extract image feature and prompt feature,
    flatten and embed voxel names, mask unused obs, etc.

    Here we just use random vectors for demo purpose.
    """
    B = 1
    obs = {
        "compass": torch.rand((B, 4), device=device),
        "gps": torch.rand((B, 3), device=device),
        "voxels": torch.randint(
            low=0, high=26, size=(B, 3 * 3 * 3), dtype=torch.long, device=device
        ),
        "biome_id": torch.randint(
            low=0, high=167, size=(B,), dtype=torch.long, device=device
        ),
        "prev_action": torch.randint(
            low=0, high=88, size=(B,), dtype=torch.long, device=device
        ),
        "prompt": torch.rand((B, 512), device=device),
        "rgb": torch.rand((B, 512), device=device),
    }
    return Batch(obs=obs)


def transform_action(action):
    """
    Map agent action to env action.
    """
    assert action.ndim == 2
    action = action[0]
    action = action.cpu().numpy()
    if action[-1] != 0 or action[-1] != 1 or action[-1] != 3:
        action[-1] = 0
    action = np.concatenate([action, np.array([0, 0])])
    return action


@torch.no_grad()
@hydra.main(config_name="conf", config_path=".", version_base="1.1")
def main(cfg):

    feature_net_kwargs = cfg.feature_net_kwargs

    feature_net = {}
    for k, v in feature_net_kwargs.items():
        v = dict(v)
        cls = v.pop("cls")
        cls = getattr(F, cls)
        feature_net[k] = cls(**v, device=device)

    feature_fusion_kwargs = cfg.feature_fusion
    feature_net = SimpleFeatureFusion(
        feature_net, **feature_fusion_kwargs, device=device
    )

    actor = MultiCategoricalActor(
        feature_net,
        action_dim=[3, 3, 4, 25, 25, 8],
        device=device,
        **cfg.actor,
    )

    mine_agent = MineAgent(
        actor=actor,
    ).to(device)

    env = CombatSpiderDenseRewardEnv(
        step_penalty=0,
        attack_reward=1,
        success_reward=10,
    )

    for i in tqdm(range(2), desc="Episode"):
        obs = env.reset()
        done = False
        pbar = tqdm(desc="Step")
        while not done:
            obs = preprocess_obs(obs)
            action = transform_action(mine_agent(obs).act)
            obs, reward, done, info = env.step(action)
            pbar.update(1)
        print(f"{i+1}-th episode ran successful!")
    env.close()


# if __name__ == "__main__":
#     main()

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = cfg.pop('ckpt_path', None)

    # print('cfg', cfg)
    # print('device', device)
    model = MineCLIP(**cfg).to(device)
    model.load_ckpt(path, strict=True)
    print("Successfully loaded ckpt")

    return model


cfg = {
    'arch': 'vit_base_p16_fz.v2.t2',
    'hidden_dim': 512,
    'image_feature_dim': 512,
    'mlp_adapter_spec': 'v0-2.t0',
    'pool_type': 'attn.d2.nh8.glusw',
    'resolution': resolution,
    'ckpt_path': 'weights/attn.pth'
}

images = arr[np.newaxis, :, :, :]
images = torch.from_numpy(images)
images = images.permute(0, 3, 1, 2)
print('images shape', images.shape)

model = main(cfg)
video = torch.stack([images], dim=0)  # video must be 5d.  [L, B, C, H, W]
print('video', video.shape)
video = video.to('cuda')
latent_obs = model.forward_image_features(video)  # output is 3d. [L, B, D]

latent_obs = latent_obs.to('cpu').detach().numpy()

# First, to resolve mapping latent observations to a mean state given this observation.
# We treat each dimension as independent of others, meaning that we get D-number of bivariate normals
# this means we have mean and variance latent_obs and state vectors of D size, and (can) have D correlation coefficients.
latent_obs_mean = np.random.randn(1, 1, 512)
latent_obs_var = np.random.randn(1, 1, 512)

state_mean = np.random.randn(1, 1, 512)
state_var = np.random.randn(1, 1, 512)

cor_coeff = np.random.randn(1, 1, 512)

posterior_state_x_mean = state_mean + cor_coeff * (latent_obs_var / state_var) * (latent_obs - latent_obs_mean)
# posterior_state_var = state_var**2 * (1 - cor_coeff**2), ignore variance for now.

