from pathlib import Path

from Senti.modules.config.config import Config
from Senti.registry import AGENTS, ENVS, GENOME


class Experiment:
    """
    Experiment class encapsulates experimental logic.

    Experiments have responsibility over initializing and building the environment and agent handler,
    and communicating between the two.
    """
    def __init__(self, config: Config):

        self.config = config
        self.rounds = config.rounds

        self.agent_handler = AGENTS.build(self.config.agent_handler.name, 
            config=self.config.agent_handler,
        )

        self.env_handler = ENVS.build(self.config.env_handler.name,
            config=self.config.env_handler
        )

        self.genome_handler = GENOME.build(self.config.genome_handler.name,
            config=self.config.genome_handler
        )

    def setup(self, prev_strategy=None):
        """
        Seperate setup stage to initialize strategies, tracking genomes, logging, etc.
        """
        strategy = self.genome_handler.strategy_factory(prev_strategy=prev_strategy)
        self.agent_handler.load(strategy)

        # TODO build env and genome handler soon

        return strategy

        
    def run(self):
        for i in range(self.rounds):

            data = self.env_handler.reset()
            # obs, rewards, terimated, truncated, infos = self.env_handler.step()
            while self.env_handler.is_valid():
                actions = self.agent_handler.forward(data['obs'])
                data = self.env_handler.step(actions)

            self.strategy=  self.strategy_factory(prev_strategy=self.strategy)

    


if __name__ == "__main__":
    default = "/mnt/e/nmmo_actinf/Senti/src/Senti/configs/base_handler.py"
    config = Config.fromfile(default)
    print('config.does_this_exist', config.does_this_exist)
    e = Experiment(config=config.experiment)
    e.setup()
    e.run()
