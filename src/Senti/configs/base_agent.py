
atomic_timestep = 2
history = 2

agent = dict(
    preferences_dim = 32,
    h_dim = [8, 8],
    z_dim = 8,
    history_size = atomic_timestep * history,
    device = "cuda"
)