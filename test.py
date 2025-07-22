import numpy as np

data = np.load("actor_rollout_log.npz")
print("Keys:", data.files)
for key in data.files:
    print(f"{key}: shape = {data[key].shape}")
