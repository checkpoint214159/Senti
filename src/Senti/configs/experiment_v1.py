_base_ = ['./base_agent.py']

# Override specific nested values
agent = dict(
    h_dim = [16, 16], # Change the resolution
    learning_rate = 1e-4 # Add a new parameter
)

# You can even use logic to modify the base
inference_steps = 100