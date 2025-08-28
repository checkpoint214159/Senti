from typing import Protocol

import torch


class GeneralLossFn(Protocol):
    def __call__(self, *args: object) -> torch.Tensor: ...

class TensorOnlyLossFn(Protocol):
    def __call__(self, *args: torch.Tensor) -> torch.Tensor: ...

class TensorAndStrLossFn(Protocol):
    def __call__(self, *args: torch.Tensor | str) -> torch.Tensor: ...


class EnergyAggregator:
    """
    Class to register losses under.
    The core functionality is to act as an aggregator, providing a function to accumulate
    all calculations using a generic instruction template
    """
    def __init__(self):
        pass

    def compute_energy(
        self,
        factors: list[tuple[GeneralLossFn, ...]]
    ):
        """
        Aggregates energies from various factors; we parse each tuple as its computation
        and aggregate the overall for the return.

        This is mostly for your reading since this method doesn't handle the exact semantic
        specifics of what you are calculating, but the VFE generaly includes:
        - KL divergence between current variational belief `qh_t` and prior `ph_t`
        - KL divergence from message from the past (if ph_from_tm1 provided)
        - KL divergence from message from the future (if qh_tp1 and ph_at_tp1 provided)
        - Negative log-likelihood between predicted and true latent observations (if provided)

        """
        total_loss = 0
        for t in factors:
            loss_func, *inputs = t
            loss = loss_func(*inputs).mean()  # -> reduce to scalar
            total_loss += loss

        return total_loss
            

    # def compute_vfe_node(
    #     self,
    #     factors: list[tuple[LossFunction, ...]]
    #     qh_t: torch.Tensor,
    #     ph_t: torch.Tensor,
    #     lat_o_pred: torch.Tensor,
    #     lat_o_true: torch.Tensor,
    #     ph_from_tm1: torch.Tensor | None = None,
    #     qh_tp1: torch.Tensor | None = None,
    #     ph_at_tp1: torch.Tensor | None = None,
    #     variance_prior: float = 1.0,
    #     variance_obs: float = 1.0
    # ) -> torch.Tensor:
    #     """
    #     Computes the variational free energy (VFE) at a specific timestep `t` for a single state node.

        
    #     Args:
    #         qh_t (torch.Tensor): Current posterior mean (e.g., hidden state) at time `t`.
    #         ph_t (torch.Tensor): Prior mean at time `t` from dynamics.
    #         lat_o_pred (torch.Tensor): Predicted latent observation at time `t`.
    #         lat_o_true (torch.Tensor): Actual latent observation at time `t`.
    #         ph_from_tm1 (torch.Tensor, optional): Prior at `t` predicted from `t-1`.
    #         qh_tm1 (torch.Tensor, optional): Variational posterior at `t-1` (unused here, for symmetry).
    #         qh_tp1 (torch.Tensor, optional): Posterior at `t+1` (used for backward KL).
    #         ph_at_tp1 (torch.Tensor, optional): Prediction of `qh_tp1` from forward model.
    #         variance_prior (float): Variance for KL computations (assumes isotropic Gaussian).
    #         variance_obs (float): Variance used for log-likelihood (assumes isotropic Gaussian).

    #     Returns:
    #         torch.Tensor: Scalar tensor representing the total variational free energy at timestep `t`.
    #     """
    #     # Observation likelihood (negative log-likelihood)
    #     energy_obs = self.log_likelihood(lat_o_true, lat_o_pred)
    #     print('energy obs mean:', energy_obs.mean())
    #     energy_states = 0.0

    #     # Message from previous timestep
    #     if ph_from_tm1 is not None:
    #         energy_states += self.kl_divergence(qh_t, ph_from_tm1)

    #     # Message from next timestep
    #     if qh_tp1 is not None and ph_at_tp1 is not None:
    #         energy_states += self.kl_divergence(qh_tp1, ph_at_tp1)

    #     # Current prior vs posterior at time t
    #     energy_states += self.kl_divergence(qh_t, ph_t)

    #     # Total variational free energy = state energy - observation energy
    #     total_energy = (energy_states.mean() - energy_obs.mean()).sum()
    #     return total_energy
    
    @staticmethod
    def kl_divergence(mean1, mean2):
        """
        Helper func to calculate kl_divergence between two isotropic gaussians with variance 1, for now
        """
        return 0.5 * (-1 + 1 + (mean1 - mean2) ** 2)  # what even. idk bro, if gaussian is isotropic multivariate, its just this?

    @staticmethod
    def log_likelihood(o_true, o_pred):
        """
        Helper func to calculate log_likeihood of pred, w.r.t truth. Equivalent to above method for isotropic gaussians with
        var 1 too, see how we can change this
        """
        # assume we have uniform multivariate gaussian
        dist = torch.distributions.Normal(o_true, 1.0)
        return dist.log_prob(o_pred)
