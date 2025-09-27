"""
Because Microsoft's malmo troublesomely has to take about 1+ minute to build a minecraft world instance,
its very annoying for testing out module functionality as we write the script.

Thus, create a minecraft instance that simply waits for a call from some client somewhere, and only takes a
step in the environment when it recieves that. Else, the environment remains in a paused state.

Argparser is used to configure this script during runtime, but defaults are set so we don't need
to always troublesomely pass in args we don't care about.
"""
import argparse

import numpy as np
import uvicorn
from fastapi import APIRouter, FastAPI

# import matplotlib.pyplot as plt
from pydantic import BaseModel

import minedojo


class Test(BaseModel):
    action: list[int]


app = FastAPI()

class MinecraftInstance:
    """
    Class that houses functionalities to call the env to do stuff and return observations.
    """

    def __init__(self, env):
        self.env = env
        self.sample_act = env.action_space.no_op()
        self.router = APIRouter()
        self.router.add_api_route("/take_step", self.take_step, methods=["POST"])


    def take_step(self, test: Test):
        action = test.action
        assert len(action) == len(self.sample_act), f'Assertion failed. Pass in a list of integers that is of size {len(self.sample_act)}'
        obs, reward, done, info = self.env.step(action)

        def convert_arrays_to_lists(obj):
            if isinstance(obj, dict):
                return {k: convert_arrays_to_lists(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays_to_lists(elem) for elem in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_arrays_to_lists(elem) for elem in obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        obs = convert_arrays_to_lists(obs)

        return {
            'obs': obs,
            'reward': reward,
            'done': done,
            'info': info,
        }



def main():

    parser = argparse.ArgumentParser(description="A persistent minecraft instance")
    # height, width. 
    parser.add_argument('-r', '--resolution', type=int, nargs=2, default=[360, 640], help='Resolution the minecraft instance will render at.')
    parser.add_argument('-t', '--task_id', type=str, default='harvest_wool_with_shears_and_sheep', help='Valid MineDojo task id.')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('-p', '--port', type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload of server on code changes.")

    args = parser.parse_args()

    task_id = args.task_id
    resolution = args.resolution

    env, prompts = instantiate_env(task_id=task_id, resolution=resolution)
    instance = MinecraftInstance(env)
    app.include_router(instance.router)

    return args, instance


def instantiate_env(task_id, resolution):
    """
    Instantiates an environment and returns it
    """
    frameSize = (resolution[0], resolution[1])
    env = minedojo.make(task_id=task_id, image_size=frameSize)
    prompts = [env.task_prompt]
    env.reset()

    return env, prompts

if __name__ == "__main__":
    args, instance = main()
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except KeyboardInterrupt:
        instance.env.close()
