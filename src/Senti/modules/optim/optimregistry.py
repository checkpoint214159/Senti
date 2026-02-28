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

    def do_functional_step(self, key, params_dict, grads_dict):
        """
        Updates parameters using externally computed gradients (e.g., from VJP).
        """
        optim = self.retrieve_optimizer(key)
        optim.zero_grad()

        # 1. Manually inject the functional gradients into the parameter objects
        for name, p in params_dict.items():
            if name in grads_dict and grads_dict[name] is not None:
                # We assign the grad attribute directly
                # .detach() ensures we don't accidentally track the grad itself
                p.grad = grads_dict[name].detach()
            else:
                # If a parameter didn't get a gradient, ensure it's None or Zero
                p.grad = None

        # 2. Run the optimizer step as usual
        optim.step()
    