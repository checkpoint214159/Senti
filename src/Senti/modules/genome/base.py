from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch import nn

from Senti.registry import GENOME


class TensorModule(nn.Module):
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.data = nn.Parameter(tensor)

@GENOME.register_module()
class BaseGenome(nn.Module):
    """
    Simple metadata driven BaseGenome class to wrap around a module. So that we can easily wrap around anything
    and call our methods for it
    Do not provide any default kwargs: force initializer to provide the state dict AND the module it is meant to be loaded into.
    This means that Genomes
    
    Also doesnt handle pathing or filenaming, that should be up to the Selector that is using us
    This forces people to subtype Genomes instead of defining them dynamically during runtime.
    """
    def __init__(self, 
        blueprint: nn.Module, 
        gene: dict):

        super().__init__()

        self.blueprint = blueprint 

        assert self.validate(gene, self.blueprint), 'Assertion failed, blueprint and state_dict passed to BaseGenome do not match.'
        self.gene = gene
    
    @staticmethod
    def validate(data: dict, blueprint: nn.Module):
        """checks if the data matches the blueprint's keys and shapes."""
        reference_sd = blueprint.state_dict()
        
        for key, ref_tensor in reference_sd.items():
            if key not in data:
                raise KeyError(f"Genome missing required key: {key}")
            
            if data[key].shape != ref_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: "
                    f"Expected {ref_tensor.shape}, got {data[key].shape}"
                )
        return True

    def load_path(self, path: str | Path, device: str = 'cuda'):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No genome found at {path}")
        
        data = torch.load(path, map_location=device)
    
        if 'state_dict' in data:
            data = data['state_dict']
            
        if self.validate(data, self.blueprint):
            self.gene = data
        return self

    @classmethod
    def of(cls, blueprint: nn.Module, data: dict):
        """factory: inits a genome from data"""
        instance = cls(blueprint)
        if instance.validate(data):
            instance.gene = data
        return instance
    
    @property
    def state_dict(self) -> dict:
        return self.gene


@GENOME.register_module()
class MutateGenome(BaseGenome):
    """
    Flexible mutation genome that supports different noise distributions
    used to mutate particular names of the state dict.
    """
    def __init__(self,
                blueprint: nn.Module,
                gene: dict | None = None,
                noise_type: str = 'gaussian',
                mutation_keys: List[str] = [],
                ):

        super().__init__(blueprint, gene)
        self.noise_type = noise_type.lower()
        self.mutation_keys = mutation_keys
        for key in self.mutation_keys:
            if key not in self.gene:
                raise NotImplementedError(f'key {key} in mutation_keys is not in self.gene dictionary.')
        
    def _generate_noise(self, base_tensor, scale):
        """Generates noise based on the selected torch distribution."""
        if self.noise_type == 'gaussian':
            return torch.randn_like(base_tensor) * scale
        elif self.noise_type == 'uniform':
            return (torch.rand_like(base_tensor) * 2 - 1) * scale
        elif self.noise_type == 'laplace':
            dist = torch.distributions.Laplace(0, scale)
            return dist.sample(base_tensor.shape).to(base_tensor.device)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def mutate(self, mutation_rate: float):
        """
        Applies mutation to the weights in self.gene.
        
        Args:
            mutation_rate: The scale of the noise.
            target_prefix: If provided, only keys starting with this string 
                           (e.g., 'habitual_head') will be mutated.
        """
        #todo: add warnings instead of erroring out here?
        with torch.no_grad(): 
            for key, weights in self.gene.items():
                if key not in self.mutation_keys:
                    continue
                elif torch.is_floating_point(weights):
                    noise = self._generate_noise(weights, mutation_rate)
                    weights.add_(noise)
                else:
                    raise NotImplementedError("Currently cannot mutate non-floating point weights")

        return self


# @dataclass
# class GenomeRecord:
#     name: str
#     path: Path
#     genome: BaseGenome
#     fitness: float
#     generation: int
#     metadata: dict = None

# TODO: outdated semantic. replaced with genome handler, todo delete this
# @SELECTOR.register_module()
# class BaseSelector:
#     def __init__(self,
#         config: DictConfig,
#         ):
#         self.config = config
#         self.root = self.config.root
        
#         self._registry: dict[str, GenomeRecord] = {}

#     def add(self, name:str, genome: BaseGenome,
#         fitness: float, generation: int, **kwargs):
#         assert name not in self._registry, 'unique names for now'
#         path = (Path(self.root) / name)
#         path.mkdir(parents=True, exist_ok=True)
#         record = GenomeRecord(
#             name=name,
#             path=path,
#             genome=genome, 
#             fitness=fitness, 
#             generation=generation, 
#             metadata=kwargs
#         )
#         self._registry[name] = record

#     def sort(self, by: str = "fitness", descending: bool = True) -> List[GenomeRecord]:
#         """Sorts the records (values) and returns a list."""
#         return sorted(
#             self._registry.values(), 
#             key=lambda x: getattr(x, by), 
#             reverse=descending
#         )
    
#     def make_filename(self, record):
#         "uses the record to create a filename"
#         return Path(record.path) / \
#             f"{record.name}_gen{record.generation}_fitness{record.fitness}.pth"
    
#     def checkpoint(self, name):
#         """
#         retrieves genome's state and saves it.
#         """
#         if name not in self._registry:
#             raise KeyError(f"Genome '{name}' not found in registry.")
        
#         record = self._registry[name]
#         fn = self.make_filename(record)
#         metadata = {
#             'generation': record.generation,
#             'fitness': record.fitness,
#             'type': type(record.genome).__name__
#         }
#         state = record.genome.state_dict()
#         overlapping_keys = set(state.keys()) & set(metadata.keys())
#         assert not overlapping_keys, 'Assertion failed, state and metadata share keys, check genome to see if saved inappropriate key names'
#         state = state | metadata
#         torch.save(state, fn)

#     def load(self, name, path):
#         """
#         loads a genome from a path, and adds it to registry with a name
#         """
#         state = torch.load(path)
#         genome_cls = state['type']
#         fitness = state['fitness']
#         generation = state['generation']

#         genome = GENOME.run(genome_cls, 'load', state)
#         self.add(
#             name=name,
#             genome=genome,
#             fitness=fitness,
#             generation=generation
#         )

#     def query(self, generation: Optional[int] = None, min_fitness: Optional[float] = None) \
#         -> List[GenomeRecord]:
#         results = list(self._registry.values())
#         if generation is not None:
#             results = [r for r in results if r.generation == generation]
#         if min_fitness is not None:
#             results = [r for r in results if r.fitness >= min_fitness]
#         return results

#     def get_best(self, n: int = 1) -> List[BaseGenome]:
#         """Returns the actual genomes of the top N performers."""
#         sorted_records = self.sort(by="fitness", descending=True)
#         return [r.genome for r in sorted_records[:n]]

#     # def save_to_csv(self, filename: str = "registry_metadata.csv"):
#     #     """Exports the metadata (minus the actual genome objects) to CSV."""
#     #     path = self.root / filename
        
#     #     data = []
#     #     for r in self._registry.values():
#     #         # Combine core stats with whatever is in metadata dict
#     #         row = {
#     #             "name": r.name,
#     #             "fitness": r.fitness,
#     #             "generation": r.generation,
#     #             "path": str(r.path)
#     #         }
#     #         if r.metadata:
#     #             row.update(r.metadata)
#     #         data.append(row)
            
#     #     pd.DataFrame(data).to_csv(path, index=False)
