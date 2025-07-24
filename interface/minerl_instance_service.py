import argparse
import numpy as np
import uvicorn
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
import gym
import minerl

class ActionRequest(BaseModel):
    action: list[int]

app = FastAPI()

class MinecraftInstance:
    """
    Wraps a MineRL environment behind a FastAPI router.
    """
    def __init__(self, env):
        self.env = env
        # MineRL’s action_space.noop() is accessed via gym spaces directly:
        self.sample_act = self.env.action_space.noop()
        self.router = APIRouter()
        self.router.add_api_route("/take_step", self.take_step, methods=["POST"])

    def take_step(self, req: ActionRequest):
        action = req.action
        assert len(action) == len(self.sample_act), (
            f"Action length must be {len(self.sample_act)}"
        )
        obs, reward, done, info = self.env.step(action)

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
        "-e", "--env", type=str,
        default="MineRLNavigateDense-v0",
        help="Gym environment name (e.g. MineRLNavigateDense-v0)"
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

    env = instantiate_env(args.env)
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
