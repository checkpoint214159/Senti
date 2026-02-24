from Senti.modules.genome.base import MutateGenome
from Senti.registry import GENOME


@GENOME.register_module()
class POMDPAgentGenome(MutateGenome):
    """
    Specific MutateGenome expecting a key of 'preference_genome', will raise
    custom errors.
    TODO merge this into a more general form of MutateGenome?
    """

    def __init__(self,
        *args,
        **kwargs,
        ):

        super().__init__(*args, **kwargs, mutation_keys=['preference_genome'])

    def state_dict(self):
        if 'preference_genome' not in self.gene:
            raise NameError('preference_genome not found in self.gene. Possibly touched something private in the Genome?')