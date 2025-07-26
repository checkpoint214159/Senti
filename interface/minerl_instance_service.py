import argparse

import gym
import numpy as np
import uvicorn
from fastapi import APIRouter, FastAPI
from gym import spaces
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from pydantic import BaseModel


class ActionRequest(BaseModel):
    action: dict[str, list[int] | list[float] | list[list[float]]]

app = FastAPI()

ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)

TARGET_ACTION_SPACE = {
    "ESC": spaces.Discrete(2),
    "attack": spaces.Discrete(2),
    "back": spaces.Discrete(2),
    "camera": spaces.Box(low=-180.0, high=180.0, shape=(2,)),
    "drop": spaces.Discrete(2),
    "forward": spaces.Discrete(2),
    "hotbar.1": spaces.Discrete(2),
    "hotbar.2": spaces.Discrete(2),
    "hotbar.3": spaces.Discrete(2),
    "hotbar.4": spaces.Discrete(2),
    "hotbar.5": spaces.Discrete(2),
    "hotbar.6": spaces.Discrete(2),
    "hotbar.7": spaces.Discrete(2),
    "hotbar.8": spaces.Discrete(2),
    "hotbar.9": spaces.Discrete(2),
    "inventory": spaces.Discrete(2),
    "jump": spaces.Discrete(2),
    "left": spaces.Discrete(2),
    "pickItem": spaces.Discrete(2),
    "right": spaces.Discrete(2),
    "sneak": spaces.Discrete(2),
    "sprint": spaces.Discrete(2),
    "swapHands": spaces.Discrete(2),
    "use": spaces.Discrete(2)
}

def validate_env(env):
    """Check that the MineRL environment is setup correctly, and raise if not"""
    for key, value in ENV_KWARGS.items():
        if key == "frameskip":
            continue
        if getattr(env.task, key) != value:
            raise ValueError(f"MineRL environment setting {key} does not match {value}")
    action_names = set(env.action_space.spaces.keys())
    if action_names != set(TARGET_ACTION_SPACE.keys()):
        raise ValueError(f"MineRL action space does match. Expected actions {set(TARGET_ACTION_SPACE.keys())}")

    for ac_space_name, ac_space_space in TARGET_ACTION_SPACE.items():
        if env.action_space.spaces[ac_space_name] != ac_space_space:
            raise ValueError(f"MineRL action space setting {ac_space_name} does not match {ac_space_space}")


class MinecraftInstance:
    """
    Wraps a MineRL environment behind a FastAPI router.
    """
    def __init__(self, env):
        validate_env(env)
        self.env = env
        # MineRL’s action_space.noop() is accessed via gym spaces directly:
        self.sample_act = self.env.action_space.noop()
        self.router = APIRouter()
        self.router.add_api_route("/take_step", self.take_step, methods=["POST"])

    def take_step(self, req: ActionRequest):
        action_dict = {
            k: np.array(v) for k, v in req.action.items()
        }
        # print('self.sample_act', self.sample_act)
        # assert set(action_dict.keys()) == set(self.sample_act.keys()), (
        #     f"Action keys must match sample_act keys: {list(self.sample_act.keys())}"
        # )

        obs, reward, done, info = self.env.step(action_dict)
        self.env.render()
        # print('obs', obs)

        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(convert(v) for v in obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        return {
            "obs": convert(obs),
            "reward": float(reward),
            "done": bool(done),
            "info": convert(info),
        }

def instantiate_env(env_name: str, **kwargs):
    """
    Create and return a MineRL Gym environment.
    """
    env = gym.make(env_name, **kwargs)
    env.reset()
    return env

def main():
    parser = argparse.ArgumentParser(
        description="Persistent MineRL instance via FastAPI"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Server host"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=8000,
        help="Server port"
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload"
    )

    args = parser.parse_args()

    # For now, only instantiate Herobraine HumanSurvival env
    # env = instantiate_env(args.env)
    # action = env.action_space.sample()
    env = HumanSurvival(**ENV_KWARGS).make()
    env.reset()
    instance = MinecraftInstance(env)
    app.include_router(instance.router)

    return args, instance

if __name__ == "__main__":
    args, instance = main()
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    except KeyboardInterrupt:
        instance.env.close()
