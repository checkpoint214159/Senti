_base_ = [
    './base_selector.py'
]

device = 'cuda'
pref_gene_dim = 32
policy_dim = 16
n_policies = 4
h_dim = [8, 8]
z_dim = 8
a_dim = 8
obs_dim = 10
state_depth = 3
hidden_dim_internal = 20
history = 5
atomic_timestep = 2
batch_size=8
inference_steps=10
param_learning_steps=2
planning_horizon=5
planning_cycles=3
discrete_step=2

nmmo_obs_encoder = dict(
    name='NmmoEncoders',
    intermediate=32,
    hidden_size=obs_dim,
    device=device
)

nmmo_obs_decoder = dict(
    name='NmmoDecoders',
    intermediate=32,
    hidden_size=obs_dim,
    device=device
)

agent = dict(
    pref_gene_dim=pref_gene_dim,
    policy_dim=policy_dim,
    n_policies=n_policies,
    h_dim=h_dim,
    z_dim=z_dim,
    a_dim=a_dim,
    obs_dim=obs_dim,
    state_depth=state_depth,
    device=device,
    history=history,
    atomic_timestep=atomic_timestep,
    batch_size=batch_size,
    inference_steps=inference_steps,
    param_learning_steps=param_learning_steps,
    planning_horizon=planning_horizon,
    planning_cycles=planning_cycles,
    discrete_step=discrete_step,

    # Nested Model Components
    nmmo_obs_encoder=nmmo_obs_encoder,
    nmmo_obs_decoder=nmmo_obs_decoder,
    
    NmmoObsAE=dict(
        encoder=nmmo_obs_encoder,
        decoder=nmmo_obs_decoder
    ),

    zh_o=dict(
        h_dim=h_dim,
        z_dim=z_dim,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim_internal
    ),

    transition_model=dict(
        h_dim=h_dim,
        z_dim=z_dim,
        a_dim=a_dim,
        external_dim=obs_dim,
        device=device,
        hidden_dim=hidden_dim_internal,
        num_GRU_layers=state_depth
    ),

    deliberative_head=dict(
        h_dim=h_dim,
        z_dim=z_dim,
        a_dim=a_dim,
        pref_gene_dim=pref_gene_dim,
        policy_dim=policy_dim
    ),

    grounding_wm=dict(
        h_dim=h_dim,
        z_dim=z_dim,
        a_dim=a_dim,
        external_dim=pref_gene_dim,
        device=device,
        hidden_dim=hidden_dim_internal,
        num_GRU_layers=state_depth
    ),

    habitual_head=dict(
        h_dim=h_dim,
        z_dim=z_dim,
        a_dim=a_dim
    ),

    pref=dict(
        h_dim=h_dim,
        z_dim=z_dim
    ),

    policy_predictor=dict(
        pref_gene_dim=pref_gene_dim,
        policy_dim=policy_dim
    )
)