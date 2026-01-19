import torch
from torch import nn

from Senti.modules.genome.base import BaseGenome, MutateGenome
from Senti.registry import GENOME


@GENOME.register_module()
class Genome(BaseGenome):
    """
    Class primarily to wrap around two genomes
    and call their methods together

    The semantic difference between them is that structural_model is a BaseGenome
    whilst the other other is a MutateGenome
    """
    def __init__(self, morphology: BaseGenome, preferences: MutateGenome):
        
        self.morphology = morphology 
        self.preferences = preferences
    
    @classmethod
    def load(cls, state:dict):
        return cls(
            morphology=BaseGenome.load(state['morphology']),
            preferences=MutateGenome.load(state['preferences']),
        )
