from functools import reduce

import torch
from torch import nn

from Senti.modules.genome.base import BaseGenome, MutateGenome
from Senti.registry import GENOME


def merge_modules(base_module, other_module):
    """merges two modules recklessly"""
    for name, module in other_module.named_children():
        if not hasattr(base_module, name):
            base_module.add_module(name, module)
        else:
            raise NameError(f"Merging modules failed: base and other module have same a overlapping attribute: {name}")
    return base_module

def merge_state_dict(base_sd, other_sd):
    for name, state in other_sd.items():
        if name not in base_sd:
            base_sd[name] = state
        else:
            raise NameError(f"Merging modules failed: base and other state dict have same a overlapping attribute: {name}")
    return base_sd



@GENOME.register_module()
class Genome(BaseGenome):
    """
    Class primarily to wrap around multiple genomes
    and call their methods together.

    Necessitates a merging of genome's blueprints and state_dicts.
    """

    def __init__(self, 
        genomes: list["BaseGenome"]):

        assert len(set([type(g) for g in genomes])) == len(genomes), \
            'Assertion failed. genomes must be a unique set of genome types, '
        blueprint = reduce(merge_modules, [g.blueprint for g in genomes])
        
        self.genomes = genomes
        super().__init__(blueprint, self.state_dict())

    def __call__(self, method_name: str, **kwargs):
        for name, g in self.genomes.items():
            method = g.getattr(method_name)
            if isinstance(method, callable):
                self.genomes[name] = method(kwargs)

    def cls_names_dict(self):
        """
        mapping of the genome class to the names it controls.
        this is useful where we want to understand the mapping of the various genome types to their
        """
        return {
            type(g): list(g.state_dict.keys()) for g in self.genomes
        }

    def state_dict(self):
        return reduce(merge_state_dict, [g.state_dict for g in self.genomes])
    
    @classmethod
    def of(self):
        """of can no longer be a classmethod, since """
