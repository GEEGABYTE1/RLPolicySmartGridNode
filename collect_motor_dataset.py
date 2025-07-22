import numpy as np
from motor_env_dreamer import MotorEnvDreamer
import tqdm

env = MotorEnvDreamer(render=False)
episodes = 50
steps_per_episode = 1000

data = {
    "obs": [],
    "action": [],
    "next_obs": [],
    "reward": []
}

for ep in tqdm.tqdm(range(episodes)):
    obs, _ = env.reset()
    for _ in range(steps_per_episode):
        action = np.random.uniform(low=-1.0, high=1.0, size=(1,))
        next_obs, reward, done, _, _ = env.step(action)

       
        norm_obs = np.array([
            obs[0] / np.pi,       # theta
            obs[1] / 10.0,        # theta_dot
            obs[2] / np.pi        # theta_target
        ])
        norm_next_obs = np.array([
            next_obs[0] / np.pi,
            next_obs[1] / 10.0,
            next_obs[2] / np.pi
        ])

  
        scaled_reward = reward / 10.0

        data["obs"].append(norm_obs)
        data["action"].append(action)
        data["next_obs"].append(norm_next_obs)
        data["reward"].append(scaled_reward)

        obs = next_obs

np.savez_compressed("motor_dataset_normalized.npz", 
                    obs=np.array(data["obs"]),
                    action=np.array(data["action"]),
                    next_obs=np.array(data["next_obs"]),
                    reward=np.array(data["reward"]))

print("✅ Normalized and scaled dataset saved to motor_dataset_normalized.npz")
