from torch import nn

from Senti.modules.genome.base import MutateGenome


class PreferenceGenome(MutateGenome):
    """
    specific implementation of genome that enforces a one-key dictionary with 
    a 1-dim tensor.
    """
    def __init__(self,
        gene: nn.Module
    ):
        super().__init__(gene)
        assert set(self.gene.keys()) == {'preferences'}, \
            'Genome must have exactly one key: "preferences".'
        
        assert self.gene['preferences'].ndim == 1, \
            'Dimension of preference tensor must be 1.'
            
        self.pref_gene_dim = self.gene['preferences'].shape[0]

        


