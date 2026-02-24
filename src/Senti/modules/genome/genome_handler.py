from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, validator

from Senti.modules.config.config import Config
from Senti.modules.genome.genome import BaseGenome
from Senti.registry import GENOME


class GenotypeStrategy(BaseModel):
    """
    dataclass to encapsulate id_genotype -> genome and id_genotype -> agents mappings.
    Does not actually handle mutation, saving, tracking, etc etc. of genomes ofc.
    """
    id_genotype: Dict[int, Optional[BaseGenome]] = Field(
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
        # schema_extra = {
        #     "example": {
        #         "id_genotype": {1: "checkpoints/team_a.pth", 2: "checkpoints/team_b.pth"},
        #         "agents": {1: [0, 1, 2, 3], 2: [4, 5, 6, 7]}
        #     }
        # }


@dataclass
class GenomeRecord:
    """
    Record of Genome.
    """
    name: str
    path: Path
    genome: BaseGenome
    fitness: float
    generation: int
    metadata: dict = None

@GENOME.register_module()
class BaseGenomeHandler:
    """
    Class to handle everything Genome related, except the raw genome itself.

    This handler has several limited responsibilities:
    1. Describe a strategy with which AgentHandler loads from
    2. House comparison and selection functionalities across its genomes. E.g genome A vs genome B,
    which has higher score?
    3. Use 2. to effect generation of 1.
    4. Call genome specific functions, e.g MutateGenome's mutation
    5. Manage and describe paths of genome checkpoints.


    Does not directly access and load from path.
    """
    def __init__(self, config: Config):

        self.config = config
        self.root = self.config.root
        self.n_teams = config.n_teams
        self.team_agents = config.team_agents

    
    def strategy_factory(self, prev_strategy: GenotypeStrategy | None = None) -> GenotypeStrategy:
        """
        Produces a GenotypeStr
        """
        if prev_strategy is None:
            strat = GenotypeStrategy(
                id_genotype={i: None for i in range(self.n_teams)},
                agents={
                    i: list(range(i * self.team_agents, (i + 1) * self.team_agents))
                    for i in range(self.n_teams)
                },
            )
        # else:


        # # inject fake checkpoint
        # # example_strat.id_genotype[0] = Path("/mnt/e/nmmo_actinf/Senti/temp/test/test_gen12_fitness0.0.pth")

        return strat



