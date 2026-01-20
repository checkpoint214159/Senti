from Senti.registry import ENVS
from Senti.modules.config.config import Config
import requests
import json


@ENVS.register_module()
class BaseEnvHandler:
    """
    Simple class to handle environment interfacing with from a environment server.
    
    TODO: define some kind of basic strategy to configure a 'means of decoding',
    i.e each envhandler, after getting from the address, checks + decodes that the observation
    is as expected.
    """

    def __init__(self, config: Config):
        self.config: Config = config
        self.addr = self.config.addr
        self.spaces = self.get_spaces()

    def _decode_response(self, data):
        raise NotImplementedError("Must implement a decode response method specific to the environment")

    def get_spaces(self):
        response = requests.get(f"{self.addr}/spaces")
        return response.json()

    def reset(self):
        response = requests.post(f"{self.addr}/reset")
        return self._decode_response(response.json())

    def step(self, action=None):  # for now no action
        response = requests.post(f"{self.addr}/step")
        return self._decode_response(response.json())
