"""
This module is meant to encapsulate the entire agent. 
Its scope contains all its components.
Its methods controls the flow of everything.
"""
import copy
import logging
import random
from pathlib import Path

import torch
from torch import nn

from mineclip import MineCLIP


def setup_logging(log_path: Path):
    """Set up logging only if not already configured."""
    if not logging.getLogger().hasHandlers():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

# Example usage
log_file = Path("logs/app.log")
setup_logging(log_file)

logging.info("Logging is set up!")

class Agent(nn.Module):
    """
    A minecraft agent built on the concepts of active inference.

    Observation cache: This is a cache of observations of previous timesteps.
    states cache: This is a cache of belief over states at previous timesteps.
        Note we do not use a seperate cache for P_states since we do Bayesian Message Passing.
        Deepcopy is created at the point of use to save P_states, then the variational distribution
        Q_s is updated via updating states cache.

    For now, both amortized and non-amortized inference will be supported, for research purposes.
    """

    def __init__(self, clip_config=None, num_policies=None, amortized_inf=True):
        # admin stuff
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # path = clip_config.pop('ckpt_path', None)
        self.amortized_inf = amortized_inf

        # observations and states cache.
        self.observation_cache = {}
        self.latent_observation_cache = {}
        self.states_cache = {}

        # encoder to encode incoming observation(s)
        # self.encoder = MineCLIP(**clip_config).to(device)
        # self.encoder.load_ckpt(path, strict=True)
        # logging.info("Successfully loaded MineCLIP encoder.")

        self.posterior_state_model = nn.Linear(2, 2)  # P(s|o)

        # maps latent observation to latent states
        self.likelihood_model = nn.Linear(2, 2) # P(o|s).

        self.preference_net = 1
        self.interpreter = 1
        self.transition_model = nn.Linear(2, 2)

        params = (
            list(self.posterior_state_model.parameters()) +
            list(self.likelihood_model.parameters()) +
            list(self.transition_model.parameters())
        )
        self.model_optimizer = torch.optim.SGD(params, lr=0.05)
                
        self.num_policies = num_policies
        self.state_dicts = {}

        self.curr_timestep = 0

    def forward(self,
                observations,
            ):
        """
        Order of execution:
            Perception:
                1. Encode current observation into a latent observation.
                2. Encode latent observation into belief about latent state at current timestep t.
            Belief updating:
                3. From existing beliefs in the belief cache, calculate VFE and do belief updates, to minimize
                    VFE
            Parameter learning:
                4. Final VFE acts as model loss to update parameters
            Planning:
                5. Sample the several policies from the new model.
                6. Within each policy, calculate EFE.
                7. Update distribution over policies
                8. Repeat 5-7 until convergence.
            Action:
                8. Sample a policy from the final distribution over policies, then take action
        Then increment timestep by 1.
        """
        # 1 - 2:
        self.percieve(observations)

        # 3 - 4:
        self.inference()
        # update parameters via vfe sum as loss.

        # 
        # converged = False
        # while not converged:
        #     for i in range(self.num_policies):
        #         pi = Q.sample  # some sample method
                
        #         # minimize vfe

        # # 8 - 10:
        # policy = Q.sample
        # action = policy.sample()

        # self.curr_timestep += 1

        # return action
    
    def kl_divergence(self, mean1, mean2):
        """
        Calculate KL divergence given two univariate gaussians, according to 
        https://math.stackexchange.com/questions/2888353/how-to-analytically-compute-kl-divergence-of-two-gaussian-distributions
        """
        return 0.5 * (-1 + 1 + (mean1 - mean2) ** 2)  # what even. idk bro, if gaussian is isotropic multivariate, its just this?

    def log_likelihood(self, o_true, o_pred):
        # assume we have uniform multivariate gaussian
        dist = torch.distributions.Normal(o_true, 1.0)
        return dist.log_prob(o_pred)

    def planning(
            self,
            policy,
        ):
        raise NotImplementedError

    def percieve(
            self,
            observations
        ):
        """
        Encodes current observation to a latent form, then encodes beliefs about current state.
        """

        # latent_observation = self.encoder.forward_image_features(observations)  # observations must be 5d. tensor.
        # q_s_t = self.posterior_state_model(latent_observation)
        
        q_s_t = self.posterior_state_model(observations)
        t = self.curr_timestep

        self.latent_observation_cache[t] = torch.nn.Parameter(observations)
        self.states_cache[t] = torch.nn.Parameter(q_s_t)

        # seperate optimizer over beliefs
        all_params = list(self.states_cache.values())
        self.beliefs_optimizer = torch.optim.SGD(all_params, lr=0.5)

    def inference(
        self,
        max_steps: int = 10000,
        update_rounds: int = 10,
        belief_lr: float = 0.001,
        vfe_tol: float = 1e-4,
        patience: int = 10
    ) -> float:
        """
        Run the inference procedure for belief updates and model parameter learning.
        Docstring was chatgpt'd lol, but i vetted stuff

        Args:
            max_steps (int): Maximum number of message passing steps to perform.
            update_rounds (int): Number of belief updates before checking VFE and (possibly) updating model parameters.
            belief_lr (float): Learning rate for belief updates (used in non-amortized inference).
            vfe_tol (float): VFE convergence threshold. If change in VFE is below this value, convergence is considered stable.
            patience (int): Number of consecutive stable VFE steps required to declare convergence in non-amortized inference.

        Returns:
            float: Final VFE after inference and (optional) learning.
        """
        # save prior beliefs as P (fixed reference)
        P_states = {k: s.clone() for k, s in self.states_cache.items()}
        prev_vfe = float('inf')
        stable_steps = 0

        for step in range(max_steps):
            # Pick a random timestep to update
            timestep = random.choice(list(self.states_cache))
            # print('CHOSEN TIMESTEP:', timestep)
            self.inference_step(timestep, P_states)  # lr scheduling comes later. test first

            # Every `update_rounds`, evaluate VFE and decide on learning or convergence
            if step % update_rounds == 0:
                current_vfe = self.compute_vfe(P_states).sum()
                P_states = {k: s.clone().detach() for k, s in P_states.items()}  # detaching is very important to ensure
                # previous computations that have been included in lossient involving current_vfe, affect future computations.
                print(f"[Step {step}] VFE = {current_vfe:.6f}")

                if self.amortized_inf:
                    # param learning for amortized case
                    self.param_learning(current_vfe)
                    pass

                else:
                    # check for VFE convergence in non-amortized case
                    if abs(prev_vfe - current_vfe) < vfe_tol:
                        stable_steps += 1
                        if stable_steps >= patience:
                            print("Converged.")
                            self.param_learning(current_vfe)
                            return current_vfe
                    else:
                        stable_steps = 0  # Reset if VFE jumped

                    prev_vfe = current_vfe

        print("Reached max inference steps.")
        return current_vfe
    
    def inference_step(
            self,
            timestep,
            P_states,
        ):
        """
        Step function to perform belief updating over cache of beliefs.

        For each step of belief updating, we isolate one node, calculate the corresponding messages
        fed to it, then update it. Messages include
            - observation message
            - prev state message
            - future state message
        """
        qs_t = self.states_cache[timestep]
        ps_t = P_states[timestep]
        
        # observation messages
        o_pred = self.likelihood_model(qs_t)
        o_true = self.latent_observation_cache[timestep]
        
        # Compute gradient of prediction error w.r.t. state  (right now simple form TODO not simple form?)
        loss_obs = self.log_likelihood(o_true, o_pred)

        # transition messages
        loss_trans = 0.0
        
        # msg from previous timestep
        if (timestep - 1) in self.states_cache:
            s_prev = self.states_cache[timestep - 1]
            s_pred_from_prev = self.transition_model(s_prev)
            loss_trans += self.kl_divergence(qs_t, s_pred_from_prev)

        # msg from next timestep
        if (timestep + 1) in self.states_cache:
            s_next = self.states_cache[timestep + 1]
            s_pred_to_next = self.transition_model(qs_t)
            loss_trans += self.kl_divergence(s_next, s_pred_to_next)

        # message from prior
        loss_prior = self.kl_divergence(qs_t, ps_t)

        # sgd loss
        total_loss = (loss_prior + loss_trans - loss_obs).sum()

        # Update current belief (grad descent)
        self.beliefs_optimizer.zero_grad()
        total_loss.backward()
        self.beliefs_optimizer.step()
        # self.states_cache[timestep] = qs_t - belief_lr * total_loss

    def param_learning(self, loss):
        """
        Simple helper method
        """
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

    # TODO: find some way to do this using cached states, instead of re-computing everything
    def compute_vfe(self, P_states):
        """
        Function to compute variational free energy. Does so over existing states and observations
        cache.

        P_states: Deepcopied overhead somewhere, of states before belief updating scheme was run (priors)
        """

        total_vfe = 0.0
        timesteps = sorted(self.states_cache.keys())

        for t in timesteps:
            q_st = self.states_cache[t]
            o_t = self.latent_observation_cache[t]

            # If not first timestep AND our inference is amortized, compute predictive prior from Q_s[t-1]
            if t - 1 in self.states_cache and self.amortized_inf:
                prior_pred = self.transition_model(self.states_cache[t - 1])
            else:  # just use initial frozen prior
                prior_pred = P_states[t]

            # KL divergence between posterior and prior (complexity cost)
            kl = self.kl_divergence(q_st, prior_pred)

            # --- Likelihood term ---
            o_pred = self.likelihood_model(q_st)
            log_lik = self.log_likelihood(o_t, o_pred)

            # --- VFE per timestep ---
            vfe_t = kl - log_lik
            total_vfe += vfe_t

        return total_vfe
