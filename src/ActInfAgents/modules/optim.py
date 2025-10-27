from torch import optim


class OptimRegistry:
    """
    Generic class to hold parameters and optimizers, and serve methods
    that call them
    """
    def __init__(self):
        self.optimizers = {}
    
    def register_optim(self,
                       key: str,
                       params: list,
                       optim_name:str='AdamW',
                       **kwargs
        ):
        try:
            optimizer_class = getattr(optim, optim_name)
            if key in self.optimizers:
                print(f'WARNING: Replacing existing key {key} in optimizer dict')
            self.optimizers[key] = optimizer_class(params, **kwargs)
            
        except AttributeError:
            raise ValueError(f"{optim_name} is not a valid optimizer in torch.optim")
        
    def retrieve_optimizer(self, key):
        assert key in self.optimizers, 'Assertion failed, tried to retrieve optimizer under nonexistent key'
        return self.optimizers[key]
        
    def do_step(self, key, loss):
        """
        Runs step on relevant parameter group. Presume .backward has not been called on loss
        """
        optim = self.retrieve_optimizer(key)
        optim.zero_grad()
        loss.backward()
        optim.step()
    