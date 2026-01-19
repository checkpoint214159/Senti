import time
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional

import torch
from pydantic import BaseModel, Field, field_validator, validator
from torch import nn

from Senti.modules.config.config import ConfigDict
from Senti.modules.genome.base import BaseGenome
from Senti.modules.utils.caches import Cache
from Senti.registry import AGENTS


class GenotypeStrategy(BaseModel):
    """
    dataclass to encapsulate genotype -> path and genotype -> agents mappings.
    Does not actually handle mutation, saving, tracking, etc etc. of genoms ofc.
    """
    id_genotype: Dict[int, Optional[Path]] = Field(
        default_factory=dict,
        description="mapping of genotype IDs to their respective checkpoint paths. Can be empty too"
    )
    
    # Maps Genotype ID (int) to a list of Agent IDs
    agents: Dict[int, List[int]] = Field(
        ..., 
        description="Mapping of genotype IDs to the list of agent IDs assigned to them."
    )

    @property
    def is_cold_start(self) -> bool:
        """indicates no chec"""
        return len(self.id_genotype) == 0
    
    @field_validator('id_genotype', mode='before')
    @classmethod
    def validate_paths(cls, v):
        if isinstance(v, dict):
            for key, path in v.items():
                if path == "": # catch empty strings that aren't quite None
                    v[key] = None
        return v

    @validator('agents')
    def validate_agents(cls, v, values):
        if 'id_genotype' in values:
            genotype_ids = set(values['id_genotype'].keys())
            assigned_ids = set(v.keys())
            
            if not assigned_ids.issubset(genotype_ids):
                missing = assigned_ids - genotype_ids
                raise ValueError(f"Genotype IDs {missing} are assigned to agents but have no checkpoint path.")
        return v

    class Config:
        # use Path / custom types
        arbitrary_types_allowed = True
        schema_extra = {
            "example": {
                "id_genotype": {1: "checkpoints/team_a.pth", 2: "checkpoints/team_b.pth"},
                "agents": {1: [0, 1, 2, 3], 2: [4, 5, 6, 7]}
            }
        }


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
    def __init__(self, config:ConfigDict, as_module: bool):
        super().__init__()
        self.config = config
        self.as_module = as_module

    def expose_genome(self) -> BaseGenome:
        raise NotImplementedError('Agent class must implement a function to expose its genome.')

    def build(self):
        """
        To support the actual construction of the BaseAgent, beyond its configuration in __init__.
        """
        raise NotImplementedError('Agent class must implement a function to build itself.')

example_strategy = {
    "id_genotype": {
        1: "/mnt/e/nmmo_actinf/Senti/temp/test/test_gen12_fitness0.0.pth",
        # 2: "/mnt/e/nmmo_actinf/Senti/temp/test/test_gen12_fitness0.0.pth",
    }
    ,
    "agents": {
        1: [0,],
        # 1: [0, 1, 2, 3, 4, ...],
        # 2: [8, 9, 10, 11, 12, ...],
    }
}


@AGENTS.register_module()
class BaseAgentHandler:
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

        self.config = config
        self.device = torch.device(config.device)

        self.agent_blueprint = AGENTS.build(
            self.config.agent_type, self.config.agent, as_module=True)



    def load(self,
        strategy: dict | GenotypeStrategy,
        mode: str = 'shared'):
        """
        Loads from a strategy object.

        This strategy object follows that of GenotypeStrategy.
        To account for dry initialization, there may be None values for genotypes.
        In that case, we just pass it on as None.

        mode='shared': Agents from the same genome share their parameters. This is more for 'team' based approaches,
            where the propagation of the source genome is the 'goal'.
        mode='diverged': Agents get clones and learn independently. This is more for 'survival' based approaches,
            where the surival of 'an instance' is the 'goal'.
        """

        self.mode = mode  # TODO dont init this here
        self.strategy = strategy
        self.agent_count = sum(len(ids) for ids in strategy.agents.values())
        unique_geno_ids = list(strategy.agents.keys())
        
        # maps [agent_id] -> [index_in_0th_dim_of_param_buffer]
        self.agent_to_geno_idx = torch.zeros(self.agent_count, dtype=torch.long, device=self.device)
        
        for buf_idx, (g_id, agent_ids) in enumerate(strategy.agents.items()):
            # In 'shared', every agent in a group points to one row (buf_idx)
            # In 'diverged', we'll handle this differently below
            self.agent_to_geno_idx[agent_ids] = buf_idx

        # Shared: Buffer size = number of Genotypes
        # Diverged: Buffer size = number of total Agents
        buffer_size = len(unique_geno_ids) if mode == "shared" else self.agent_count
        
        # populate the Parameter Buffer
        blueprint_params = dict(self.agent_blueprint.named_parameters())
        self._param_buffer.clear() # Reset existing params

        for name, param in blueprint_params.items():
            repeat_dims = (buffer_size,) + (1,) * param.ndim
            init_tensor = param.data.clone().repeat(*repeat_dims)
            
            # Apply Checkpoints
            for buf_idx, g_id in enumerate(unique_geno_ids):
                path = strategy.id_genotype.get(g_id)
                if path and path.exists():
                    ckpt = torch.load(path, map_location=self.device)
                    weights = ckpt[name]
                    
                    if mode == "shared":
                        init_tensor[buf_idx] = weights
                    else:
                        # in diverged mode, all agents in this genotype get these weights
                        target_agents = strategy.agents[g_id]
                        init_tensor[target_agents] = weights
            
            # register the final parameter post-loading and allocating
            safe_name = name.replace('.', '_')
            self._param_buffer[safe_name] = nn.Parameter(init_tensor)

    @property
    def merged_params(self):
        start = time.time()
        params = {}
        for safe_name, param_data in self._param_buffer.items():
            orig_name = safe_name.replace('_', '.')
            
            if self.mode == "shared":
                # indexing creates the virtual population batch [agent_count, ...]
                # gradients flow back to the unique rows in _param_buffer
                params[orig_name] = param_data[self.agent_to_geno_idx]
            else:
                params[orig_name] = param_data
        
        print('time taken for property merged_params', time.time() - start)
        return params

