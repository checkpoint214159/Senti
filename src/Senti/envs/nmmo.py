import pufferlib.pytorch
from .utils import decode_base64, decode_torch_dtype
from .base import BaseEnvHandler
from Senti.registry import ENVS


@ENVS.register_module()
class NMMOEnvHandler(BaseEnvHandler):
    """
    """

    def _decode_response(self, data: dict) -> dict:  # for now no action
        torch_dtype = decode_torch_dtype(data['view_code'])
        obs_data = decode_base64(data['data'], self.device)

        torch_observation = pufferlib.pytorch.nativize_tensor(obs_data, torch_dtype)
        data['obs'] = torch_observation
        return data
    
    def is_valid(self):
        print('DEMO VALID FUNCTION TODO MAKE IT ACTUALLY CHECK ENV')
        return True