_base_ = [
    './base_selector.py'
]

# meta shared hyperparams
device = 'cuda'
atomic_timestep = 2
gene_dim = 32
policy_dim = 16
n_policies = 4
h_dim = [8, 8]
z_dim = 8
a_dim = 8
obs_dim = 10
state_depth = 3
hidden_dim_internal = 20

n_teams = 8
team_agents = 1

play_rounds = 8


selector = dict(
    root="/mnt/e/nmmo_actinf/Senti/temp",
    seed=dict(
        gene_dim=gene_dim
    )
)


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
    atomic_timestep=atomic_timestep,
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
        gene_dim=gene_dim,
        policy_dim=policy_dim
    ),
    grounding_wm=dict(
        h_dim=h_dim,
        z_dim=z_dim,
        a_dim=a_dim,
        external_dim=gene_dim,
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
        gene_dim=gene_dim,
        policy_dim=policy_dim
    )
)

agent_handler = dict(
    name='POMDPAgentHandler',
    agent_type='Agent',
    device='cuda',
    gene_dim=gene_dim,
    batch_size=n_teams * team_agents,
    inference_steps=25,
    param_learning_steps=5,
    atomic_timestep=atomic_timestep,
    discrete_step=2,
    history=2,
    planning_horizon=5,
    planning_steps=4,
    agent=agent,
)

env_handler = dict(
    name='NMMOEnvHandler',
    addr='http://localhost:8000'
)

experiment = dict(
    rounds=play_rounds,
    agent_handler=agent_handler,
    env_handler=env_handler,
)





