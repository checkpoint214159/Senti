from Senti.modules.config.config import Config

# Load the experiment file
cfg = Config.fromfile('./experiment_v1.py')

# Accessing values with dot-notation
print(cfg.agent.h_dim)        # [16, 16] (Overridden)
print(cfg.agent.history_size) # 4 (Inherited from base calculation)
print(cfg.inference_steps)    # 100