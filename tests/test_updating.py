"""
Here we do integration testing for anything concerning updating, whether it be beliefs,
or the actual model parameters themselves.
"""
import torch

from ActInfAgents.modules.loss import Loss
from ActInfAgents.modules.optim import OptimRegistry
from ActInfAgents.modules.state import StateNode

from .conftest import default_transition_model


def test_h_updating(default_transition_model):
    """
    Flow:
        - TransitionModel initial state -> prior h -> Wrap around Optim
        - Randomly generate h -> Posterior h
        - Calculate KL loss, backprop, step in optim
        - Assert difference in previous posterior and new posterior
    """
    optim_registry = OptimRegistry()

    transition_model = default_transition_model
    ini_h_states = transition_model.initial_state(batch_size=1)
    _, h_states = StateNode.unpack_mask_states(ini_h_states)

    prior_h_statenode = StateNode(
        h_state=h_states,
        z_dim=1,
        a_dim=1,
    )

    kwargs = dict(lr=0.005)
    optim_registry.register_optim('test_h_states',
            list(prior_h_statenode.h_modules.parameters()),
            optim_name='SGD',
            **kwargs
        )
    old_h_params = prior_h_statenode.get_h_states()
    old_h_params = StateNode.flatten(old_h_params)

    new_posterior_h = [torch.randn_like(h_thing) for h_state in h_states for h_thing in h_state]
    new_posterior_h = torch.stack(new_posterior_h)

    energy_calc = Loss()
    loss = energy_calc.kl_divergence(new_posterior_h, old_h_params).mean()
    optim_registry.do_step('test_h_states', loss)

    new_h_params = StateNode.flatten(prior_h_statenode.get_h_states())
    assert not torch.equal(old_h_params, new_h_params)


    
# def test_param_updating(default_transition_model):
#     """
#     Flow:
#         - wrap transition model in optim
#         - TransitionModel initial state -> prior h
#         - Randomly generate h -> Posterior h
#         - Calculate KL loss, backprop, step in optim
#         - Assert difference in previous posterior and new posterior
#     """
#     optim_registry = OptimRegistry()
#     transition_model = default_transition_model

#     kwargs = dict(lr=0.005)
#     optim_registry.register_optim('test_param_updating',
#             list(transition_model.parameters()),
#             optim_name='SGD',
#             **kwargs
#         )
#     old_transition_model_params = list(transition_model.parameters())

#     ini_h_states = transition_model.initial_state(batch_size=1)
#     _, h_states = StateNode.unpack_mask_states(ini_h_states)

#     prior_h_statenode = StateNode(
#         h_state=h_states,
#         z_dim=1,
#         a_dim=1,
#     )

#     old_h_params = prior_h_statenode.get_h_states()
#     old_h_params = StateNode.flatten(old_h_params)

#     new_posterior_h = [torch.randn_like(h_thing) for h_state in h_states for h_thing in h_state]
#     new_posterior_h = torch.stack(new_posterior_h)

#     energy_calc = Loss()
#     loss = energy_calc.kl_divergence(new_posterior_h, old_h_params).mean()
#     optim_registry.do_step('test_param_updating', loss)

#     new_transition_model_params = list(transition_model.parameters())
#     [(
#         print(old),
#         print(new),
#         print('equal??'), print(torch.equal(old, new))
#     ) for (old, new) in zip(
#         old_transition_model_params,
#         new_transition_model_params)
#     ]
#     assert not any(not torch.equal(
#         old,
#         new
#     ) for (old, new) in zip(
#         old_transition_model_params,
#         new_transition_model_params)
#     )

