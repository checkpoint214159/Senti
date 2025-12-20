import torch
from torch import nn
from Senti.modules.genome.base import BaseGenome, MutateGenome


class Genome:
    """
    Class primarily to wrap around two genomes
    and call their methods together

    The semantic difference between them is that structural_model is a BaseGenome
    whilst the other other is a MutateGenome
    """
    def __init__(self, structural_gene: BaseGenome, habitual_genome: MutateGenome):
        
        self.structural_gene = structural_gene 
        self.habitual_genome = habitual_genome 
        
        self.fitness = 0.0
        self.generation = 0

    def mutate(self, mutation_rate=0.01):
        """
        only mutates genome
        """
        noise = torch.randn_like(self.habitual_prior) * mutation_rate
        self.habitual_prior += noise

    def save(self, path):
        state = {
            'structural_state_dict': self.structural_model.state_dict(),
            'habitual_prior': self.habitual_prior,
            'metadata': {'gen': self.generation, 'fitness': self.fitness}
        }
        torch.save(state, path)