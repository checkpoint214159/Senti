"""
This module is meant to encapsulate the entire agent. 
Its scope contains all its components.
Its methods controls the flow of everything.
"""
import random
import torch
from torch import nn
from mineclip import MineCLIP

import logging
from pathlib import Path

def setup_logging(log_path: Path):
    """Set up logging only if not already configured."""
    if not logging.getLogger().hasHandlers():
        log_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler()  # optional: also logs to stdout
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
    Q_States cache: This is a cache of belief over states at previous timesteps.
        Note we do not use a seperate cache for P_states since we do Bayesian Message Passing
    """

    def __init__(self, clip_config, num_policies):
        # admin stuff
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = clip_config.pop('ckpt_path', None)

        # observations and states cache.
        self.observation_cache = {}
        self.latent_observation_cache = {}
        self.states_cache = {}

        # encoder to encode incoming observation(s)
        self.encoder = MineCLIP(**clip_config).to(device)
        self.encoder.load_ckpt(path, strict=True)
        logging.info("Successfully loaded MineCLIP encoder.")

        self.posterior_state_model = 1  # P(s|o)

        # maps latent observation to latent states
        self.likelihood_model = 1 # P(o|s).

        self.preference_net = 1
        self.interpreter = 1
        self.transition_model = 1
        
        self.num_policies = num_policies
        self.state_dicts = {}

        self.curr_timestep = 0

    def forward(self,
                observations,
                target,
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
        self.update_beliefs()
        # update parameters via vfe sum as loss.

        # 
        converged = False
        while not converged:
            for i in range(self.num_policies):
                pi = Q.sample  # some sample method
                
                # minimize vfe

        # 8 - 10:
        policy = Q.sample
        action = policy.sample()

        self.curr_timestep += 1

        return action
    
    def kl_divergence(self, mean1, mean2):
        """
        Calculate KL divergence given two univariate gaussians, according to 
        https://math.stackexchange.com/questions/2888353/how-to-analytically-compute-kl-divergence-of-two-gaussian-distributions
        """
        return 0.5 * (-1 + 1 + (mean1 - mean2) ** 2).sum()  # what even. idk bro, if gaussian is isotropic multivariate, its just this?
    
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

        latent_observation = self.encoder.forward_image_features(observations)  # observations must be 5d. tensor.
        q_s_t = self.likelihood_model(latent_observation)
        t = self.curr_timestep

        self.states_cache[t] = q_s_t

    def update_beliefs(
        self,
        max_steps=1000,
        vfe_eval_steps=10,
        learning_rate=0.1,
        vfe_tol=1e-4,
        patience=10
    ):
        """
        Runs belief updating scheme. For max_steps, selects a timestep node and runs one step
        of belief updates, then every vfe_eval_steps, calculates VFE to see if stuff has converged.

        """
        prev_vfe = float('inf')
        stable_steps = 0

        for step in range(max_steps):
            timestep = random.choice(list(self.states_cache))
            self.update_beliefs_step(
                timestep, learning_rate
            )
            if step % vfe_eval_steps == 0:
                current_vfe = self.compute_vfe()
                print(f"[Step {step}] VFE = {current_vfe:.6f}")
            
            # primitive convergence implementation
            if abs(prev_vfe - current_vfe) < vfe_tol:
                stable_steps += 1
                if stable_steps >= patience:
                    print("Converged.")
                    return current_vfe
            else:
                stable_steps = 0

            prev_vfe = current_vfe
        
        return current_vfe
    
    def update_beliefs_step(
            self,
            timestep,
            learning_rate,
        ):
        """
        Step function to perform belief updating over cache of beliefs.

        For each step of belief updating, we isolate one node, calculate the corresponding messages
        fed to it, then update it. Messages include
            - observation message
            - prev state message
            - future state message

        Total gradient times lr = update.
        """
        s_t = self.states_cache[timestep]
        
        # ---- Observation Message ----
        # Predict observation from current state
        o_pred = self.likelihood_model(s_t)
        o_true = self.latent_observation_cache[timestep]
        
        # Compute gradient of prediction error w.r.t. state
        grad_obs = (o_pred - o_true)  # You could add .detach() if needed

        # ---- Transition Messages ----
        grad_trans = 0.0
        
        # msg from previous timestep
        if (timestep - 1) in self.states_cache:
            s_prev = self.states_cache[timestep - 1]
            s_pred_from_prev = self.transition_model(s_prev)
            grad_trans += (s_t - s_pred_from_prev)

        # msg from next timestep
        if (timestep + 1) in self.states_cache:
            s_next = self.states_cache[timestep + 1]
            s_pred_to_next = self.transition_model(s_t)
            grad_trans += (s_pred_to_next - s_next)

        # ---- Belief Update ----
        # Total gradient: from observation + transitions
        total_grad = grad_obs + grad_trans

        # Update current belief (gradient descent)
        self.states_cache[timestep] = s_t - learning_rate * total_grad

 
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal

# x = np.linspace(0, 5, 10, endpoint=False)
# y = multivariate_normal.pdf(x, mean=2.5, cov=0.5)
# print(y)   # --> p(x)

# fig1 = plt.figure()
# ax = fig1.add_subplot(111)
# ax.plot(x, y)
# plt.savefig('p(x).png')

# x, y = np.mgrid[-1:1:.01, -1:1:.05]
# print('x and y:')
# print(x, x.shape)
# print(y, y.shape)
# pos = np.dstack((x, y))
# print('pos', pos, pos.shape)
# rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 2.0]])  # mean, covar
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# ax2.contourf(x, y, rv.pdf(pos))
# print('rv.pdf pos', rv.pdf(pos))
# plt.savefig('waht3.png')

















