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
    """

    def __init__(self,
        config: ConfigDict,
    ):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self._param_buffer = nn.ParameterDict()
        self._param_names = []
        self.agent_blueprint = AGENTS.build(
            self.config.agent_type, self.config.agent, as_stateful=False, preference_genome=None)
        
    def init_strategy_params(self,
        strategy: GenotypeStrategy,
        named_params: dict,
        mode: str = 'diverged'):

        for buffer_idx, (g_id, agent_ids) in enumerate(strategy.agents.items()):
            # In 'shared', every agent in a group points to one row (buffer_idx)
            # In 'diverged', we'll handle this differently below
            self.agent_to_geno_idx[agent_ids] = buffer_idx

        # Shared: Buffer size = number of Genotypes
        # Diverged: Buffer size = number of total Agents
        buffer_size = len(strategy.agents.keys()) if mode == "shared" else self.agent_count

        # TODO: inefficient for looping. make it vectorised?
        tmp_params = {}
        for idx, name in enumerate(self._param_names):
            param = named_params[name]
            repeat_dims = (buffer_size,) + (1,) * param.ndim
            tmp_params[name] = param.data.clone().repeat(*repeat_dims)   

        return tmp_params

    def load_from_genome(self, genome: BaseGenome, tmp_params: dict, buffer_idx: int | list[int]):
        ckpt: dict = genome.state_dict()

        for name, param in tmp_params.items():
            weights = ckpt[name]
            
            tmp_params[name][buffer_idx] = weights

        return tmp_params

    def load(self,
        strategy: GenotypeStrategy,
        mode: str = 'diverged'):
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

        self.mode = mode  # TODO dont init this here
        self.strategy = strategy
        self.agent_count = sum(len(ids) for ids in strategy.agents.values())
        
        # maps [agent_id] -> [index_in_0th_dim_of_param_buffer]
        self.agent_to_geno_idx = torch.zeros(self.agent_count, dtype=torch.long, device=self.device)
        unique_geno_ids = list(strategy.agents.keys())
        
        blueprint_params = dict(self.agent_blueprint.named_parameters())
        self._param_names = list(blueprint_params.keys())
        self._param_buffer.clear() # Reset existing params

        tmp_params = self.init_strategy_params(
            strategy=strategy, mode=mode, named_params=blueprint_params)
        
        for g_id in unique_geno_ids:
            genome: BaseGenome | None = strategy.id_genotype.get(g_id)
            
            if genome is not None:
                buffer_idx = g_id if mode == "shared" else strategy.agents[g_id]
                self.load_from_genome(
                    genome=genome, tmp_params=tmp_params, buffer_idx=buffer_idx)
                
        for idx, name in enumerate(self._param_names):
            self._param_buffer[str(idx)] = nn.Parameter(tmp_params[name])

        self.to(self.device)

        # test = super().state_dict()

    @property
    def merged_params(self):
        start = time.time()
        params = {}
        for idx, name in enumerate(self._param_names):
            param_data = self._param_buffer[str(idx)]
            
            if self.mode == "shared":
                # indexing creates the virtual population batch [agent_count, ...]
                # gradients flow back to the unique rows in _param_buffer
                params[name] = param_data[self.agent_to_geno_idx]
            else:
                params[name] = param_data
        
        print('time taken for property merged_params', time.time() - start)
        return params

