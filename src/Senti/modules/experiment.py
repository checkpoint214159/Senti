from Senti.registry import AGENTS
from Senti.modules.agents.base import GenotypeStrategy

from Senti.modules.config.config import Config


class Experiment:
    """
    Experiment class encapsulates experimental logic.

    Experiments have responsibility over initializing and building the environment and agent handler,
    and communicating between the two.
    """
    def __init__(self, config: Config):

        self.config = config
        self.rounds = config.play_rounds
        self.n_teams = config.n_teams
        self.team_agents = config.team_agents
        
        # self.agent_handler_config 
        # self.env_handler_config

        self.agent_handler = AGENTS.build('POMDPAgentHandler', 
            config=self.config.handler,
        )

        self.env_handler = None  # toy value for now
        self.genome_handler = None  # TODO create a class for this

    def setup(self, prev_strategy=None):
        """
        Seperate setup stage to initialize strategies, tracking genomes, logging, etc.
        """
        strategy = self.strategy_factory(prev_strategy=prev_strategy)
        self.agent_handler.load(strategy)

        # TODO build env and genome handler soon

        return strategy

        
    def run(self):
        for i in range(self.rounds):

            obs = self.env_handler.reset()
            # obs, rewards, terimated, truncated, infos = self.env_handler.step()
            while self.env_handler.is_valid():
                actions = self.agent_handler.forward(obs)
                obs, rewards, terimated, truncated, infos = self.env_handler.step(actions)

            
            self.strategy=  self.strategy_factory(prev_strategy=self.strategy)

    def strategy_factory(self, prev_strategy: GenotypeStrategy | None = None):
        """
        currently only inits an empty strategy. TODO make it generalise to groups that
        aren't of even size?
        """

        example_strat = GenotypeStrategy(
            id_genotype={i: None for i in range(self.n_teams)},
            agents={
                i: list(range(i * self.team_agents, (i + 1) * self.team_agents))
                for i in range(self.n_teams)
            },
        )

        # inject fake checkpoint
        example_strat.id_genotype[0] = "/mnt/e/nmmo_actinf/Senti/temp/test/test_gen12_fitness0.0.pth"

        return example_strat
    


if __name__ == "__main__":
    default = "/mnt/e/nmmo_actinf/Senti/src/Senti/configs/base_handler.py"
    config = Config.fromfile(default)
    print('config.does_this_exist', config.does_this_exist)
    e = Experiment(config=config)
    e.run()
