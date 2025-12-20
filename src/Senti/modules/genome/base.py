from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from omegaconf.dictconfig import DictConfig

from Senti.registry import GENOME, SELECTOR


@GENOME.register_module()
class BaseGenome:
    """
    Simple metadata driven BaseGenome class to wrap around a module. So that we can easily wrap around anything
    and call our methods for it
    Do not provide any default kwargs: force initializer to provide everything
    Also doesnt handle pathing or filenaming, that should be up to the Selector that is using us
    """
    def __init__(self,
        gene: torch.nn.Module,
    ):
        self.gene = gene 

    # def make_filename(self):
    #     return self.path / f"name_gen{self.generation}_fitness{self.fitness}.pth"

    def save(self, path, metadata={}):
        # fn = self.make_filename()
        state = {
            'gene': self.gene.state_dict(),
            'metadata': metadata
        }
        # print('fn?', fn)
        torch.save(state, path)

    @classmethod
    def load(cls, path, gene_architecture):
        """
        TODO not updated this yet
        path: path to .pth file
        gene_architecture: The instantiated ModuleDict/Module 
                           ready to receive weights.
        """
        checkpoint = torch.load(path)
        
        gene_architecture.load_state_dict(checkpoint['state_dict'])
        generation =  checkpoint['metadata']['gen']
        fitness = checkpoint['metadata']['fitness']
        
        instance = cls(gene_architecture)
        
        instance.generation = checkpoint['metadata']['gen']
        instance.fitness = checkpoint['metadata']['fitness']
        
        return instance


@GENOME.register_module()
class MutateGenome(BaseGenome):
    """
    Flexible mutation genome that supports different noise distributions
    and targets specific 'habitual' parameters.
    """
    def __init__(self,
                 noise_type: str = 'gaussian', 
                 **kwargs):
        super().__init__(**kwargs)
        self.noise_type = noise_type.lower()
        
    def _generate_noise(self, base_tensor, scale):
        """Generates noise based on the selected torch distribution."""
        if self.noise_type == 'gaussian':
            return torch.randn_like(base_tensor) * scale
        elif self.noise_type == 'uniform':
            return (torch.rand_like(base_tensor) * 2 - 1) * scale
        elif self.noise_type == 'laplace':
            # Useful for "bursty" evolution / occasional large mutations
            dist = torch.distributions.Laplace(0, scale)
            return dist.sample(base_tensor.shape).to(base_tensor.device)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def mutate(self, mutation_rate=0.01):
        """
        Applies mutation to gene.
        """
        with torch.no_state(): # Mutation shouldn't be tracked for gradients
                noise = self._generate_noise(self.gene.habitual_prior, mutation_rate)
                self.gene.habitual_prior.add_(noise)


@dataclass
class GenomeRecord:
    name: str
    path: Path
    genome: BaseGenome
    fitness: float
    generation: int
    metadata: dict = None


@SELECTOR.register_module()
class BaseSelector:
    def __init__(self,
        config: DictConfig,
        ):
        self.config = config
        self.root = self.config.root
        
        self._registry: dict[str, GenomeRecord] = {}

    def add(self, name:str, genome: BaseGenome,
        fitness: float, generation: int, **kwargs):
        path = (Path(self.root) / name)
        path.mkdir(parents=True, exist_ok=True)
        record = GenomeRecord(
            name=name,
            path=path,
            genome=genome, 
            fitness=fitness, 
            generation=generation, 
            metadata=kwargs
        )
        self._registry[name] = record

    def sort(self, by: str = "fitness", descending: bool = True) -> List[GenomeRecord]:
        """Sorts the records (values) and returns a list."""
        return sorted(
            self._registry.values(), 
            key=lambda x: getattr(x, by), 
            reverse=descending
        )
    
    def make_filename(self, record):
        "uses the record to create a filename"
        return Path(record.path) / \
            f"{record.name}_gen{record.generation}_fitness{record.fitness}.pth"
    
    def checkpoint(self, name):
        """
        calls the named Genome's save functionality
        """
        if name not in self._registry:
            raise KeyError(f"Genome '{name}' not found in registry.")
        
        record = self._registry[name]
        metadata = {
            'generation': record.generation,
            'fitness': record.fitness,
        }
        record.genome.save(self.make_filename(record), metadata)

    def query(self, generation: Optional[int] = None, min_fitness: Optional[float] = None) \
        -> List[GenomeRecord]:
        results = list(self._registry.values())
        if generation is not None:
            results = [r for r in results if r.generation == generation]
        if min_fitness is not None:
            results = [r for r in results if r.fitness >= min_fitness]
        return results

    def get_best(self, n: int = 1) -> List[BaseGenome]:
        """Returns the actual genomes of the top N performers."""
        sorted_records = self.sort(by="fitness", descending=True)
        return [r.genome for r in sorted_records[:n]]

    def save_to_csv(self, filename: str = "registry_metadata.csv"):
        """Exports the metadata (minus the actual genome objects) to CSV."""
        path = self.root / filename
        
        data = []
        for r in self._registry.values():
            # Combine core stats with whatever is in metadata dict
            row = {
                "name": r.name,
                "fitness": r.fitness,
                "generation": r.generation,
                "path": str(r.path)
            }
            if r.metadata:
                row.update(r.metadata)
            data.append(row)
            
        pd.DataFrame(data).to_csv(path, index=False)
