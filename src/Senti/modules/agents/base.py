import time
from abc import ABC

import torch
from torch import nn

from Senti.modules.config.config import ConfigDict
from Senti.modules.genome.base import BaseGenome
from Senti.modules.genome.genome_handler import GenomeRecord, GenotypeStrategy
from Senti.modules.utils.caches import Cache
from Senti.registry import AGENTS


@AGENTS.register_module()
class BaseAgent(ABC, nn.Module):
    """
    Agents encapsulate configuration parameters and the actual nn.Module objects.
    Configuration parameters include things like model size / architecture, from
    which we retrieve + build from the registry.

    This can be constructed either as an agent with its own cache of beliefs, optimizers, etc,
    or can be held as a mere module to encapsulate the various other things it uses.

    The former should be done during development, whilst treating it as a module
    renders the logic of 'what to do with agent modules' to a different component, like an AgentHandler,
    which is usually what is best for a 'full run', during experimentation and evaluation
    """
    def __init__(self, config:ConfigDict, as_stateful: bool):
        super().__init__()
        self.config = config
        self.as_stateful = as_stateful

    def expose_genome(self) -> BaseGenome:
        raise NotImplementedError('Agent class must implement a function to expose its genome.')

    def build(self):
        """
        To support the actual construction of the BaseAgent, beyond its configuration in __init__.
        """
        raise NotImplementedError('Agent class must implement a function to build itself.')


@AGENTS.register_module()
class BaseAgentHandler(nn.Module):
    """
    AgentHandlers handles the definition of what agent type to use,
    what methods to run, loading + saving to checkpoint, etc, basically the logic
    of what to do with these agents.

    The key is that we perform synchronous mapping, in other words according to some
    well structured strategy dict, we can form a 'batch module' with slices conforming
    to the strategy.

    Strategy itself defines a mapping between Genomes, the path to load them from, and the agents that use them.
    We use path to build Genome, then load in the various semantics into the agent.

    Also comes with the ability to check against a dict of protected attributes meant to be only inited at runtime,
    specifically to optimize the agent. Pumps out a custom error if so
    """

    _OPTIM_ATTRS = {}  # placeholder class attribute to define attributes inited only at runtime

    def __getattr__(self, name):
        if name in self._OPTIM_ATTRS:
            backing_name = self._OPTIM_ATTRS[name]
            value = getattr(self, backing_name, None)
            
            if value is None:
                raise NotImplementedError(
                    f"Error: '{name}' not initialized. "
                    f"Did you mean to load this handler with 'optim=True'?"
                )
            return value

        return super().__getattr__(name)

    def __init__(self,
        config: ConfigDict,
    ):
        """
        Note: seperate param names from the dict that holds them, since raw names violate the particular
        delicate requirements of a ParameterDict (not having dots in the names)

        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self._param_buffer = nn.ParameterDict()
        self._param_names = []
        self.agent_blueprint = AGENTS.build(
            self.config.agent_type, self.config.agent, as_stateful=False, preference_genome=None)
        self.mode = self.config.mode
        assert self.mode in ['diverged', 'shared']
        
    def init_params_from_strategy(self,
        strategy: GenotypeStrategy,
        named_params: dict,
        ) -> dict:
        """
        from original param shapes creates buffers of appropriate size.

        Basically, we pad an additional dimension at dim=0, for the 'population' dim.
        If parameters are shared between teammates, population dim is equal to number of teams.
        If not, we diverge and each get their own. So far this case I will not yet handle for, because
        its quite challenging and i havent figured out the semantics of saving and loading yet.
        """

        for buffer_idx, (g_id, agent_ids) in enumerate(strategy.agents.items()):
            # In 'shared', every agent in a group points to one row (buffer_idx)
            # In 'diverged', we'll handle this differently below
            self.agent_to_geno_idx[agent_ids] = buffer_idx

        # Shared: Buffer size = number of Genotypes
        # Diverged: Buffer size = number of total Agents
        buffer_size = len(strategy.agents.keys()) if self.mode == "shared" else self.agent_count

        # TODO: inefficient for looping. make it vectorised?
        spliced_params = {}
        assert len(self._param_names) != 0, 'Assertion failed, param_names is empty, so the init_params_from_strategy method would have failed' \
            'if your strategy was valid. Check that you called load() before this?'
        
        for idx, name in enumerate(self._param_names):
            param = named_params[name]
            repeat_dims = (buffer_size,) + (1,) * param.ndim
            spliced_params[name] = param.data.clone().repeat(*repeat_dims)   

        return spliced_params

    def load_from_genome(self,
        genome: BaseGenome,
        spliced_params: dict,
        buffer_idx: list[int]
    ):
        ckpt: dict = genome.state_dict()

        for name, param in spliced_params.items():
            weights = ckpt[name]
            
            spliced_params[name][buffer_idx] = weights

        return spliced_params

        
    def load(self,
        strategy: GenotypeStrategy,
        ):
        """
        Loads from a strategy object.

        This strategy object follows that of GenotypeStrategy.
        To account for dry initialization, there may be None values for genotypes.
        In that case, we just pass it on as None.

        mode='shared': Agents from the same genome share their parameters. This is more for 'team' based approaches,
            where the propagation of the source genome is the 'goal'.
        mode='diverged': Agents get clones and learn independently. This is more for 'survival' based approaches,
            where the surival of 'an instance' is the 'goal'.
        TODO ^ ai generated docstring do it properly
        """
        self.strategy = strategy
        self.agent_count = sum(len(ids) for ids in strategy.agents.values())
        
        # maps [agent_id] -> [index_in_0th_dim_of_param_buffer]
        self.agent_to_geno_idx = torch.zeros(self.agent_count, dtype=torch.long, device=self.device)
        unique_geno_ids = list(strategy.agents.keys())
        
        blueprint_params = dict(self.agent_blueprint.named_parameters())
        self._param_names = list(blueprint_params.keys())
        self._param_buffer.clear() # ensure clean state

        spliced_params = self.init_params_from_strategy(
            strategy=strategy, named_params=blueprint_params)
        
        for g_id in unique_geno_ids:
            genome: BaseGenome | None = strategy.id_genotype.get(g_id)

            if genome is not None:
                buffer_idx = [g_id] if self.mode == "shared" else strategy.agents[g_id]
                self.load_from_genome(
                    genome=genome, spliced_params=spliced_params, buffer_idx=buffer_idx)
                
        for idx, name in enumerate(self._param_names):
            buffer_name = name.replace('.', '/')
            self._param_buffer[buffer_name] = nn.Parameter(spliced_params[name])

        self.to(self.device)

    def get_source_params(self, module_substrings: list[str]):
        return [param for name, param in self._param_buffer.items() 
                if any(mod in name for mod in module_substrings)]

    @property
    def merged_params(self):
        start = time.time()
        params = {}
        print('self.agent_to_geno_idx', self.agent_to_geno_idx)
        for idx, name in enumerate(self._param_names):
            buffer_name = name.replace('.', '/')
            param_data: nn.Parameter = self._param_buffer[buffer_name]
            
            if self.mode == "shared":
                # indexing creates the virtual population batch [agent_count, ...]
                # gradients flow back to the unique rows in _param_buffer
                params[name] = param_data[self.agent_to_geno_idx]
            else:
                params[name] = param_data
        print('time taken for property merged_params', time.time() - start)

        return params

