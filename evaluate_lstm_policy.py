import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from motor_env_dreamer import MotorEnvDreamer
from lstm_policy import LSTMPolicy

# Create output directory
os.makedirs("images", exist_ok=True)

# --- Load trained LSTM policy ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMPolicy(input_dim=3, hidden_dim=64).to(device)
model.load_state_dict(torch.load("lstm_actor.pt", map_location=device))
model.eval()

# --- Environment setup ---
env = MotorEnvDreamer(render=False)
obs_seq = []
obs, _ = env.reset()

theta_log = []
theta_dot_log = []
target_log = []
action_log = []
reward_log = []


seq_len = 10
for t in range(1000):

    norm_obs = np.array([
        obs[0] / np.pi,
        obs[1] / 10.0,
        obs[2] / np.pi
    ], dtype=np.float32)
    
    obs_seq.append(norm_obs)
    if len(obs_seq) < seq_len:
        action = np.array([0.0])
    else:
        obs_tensor = torch.tensor([obs_seq[-seq_len:]], dtype=torch.float32).to(device)
        with torch.no_grad():
            action = model(obs_tensor).cpu().numpy()[0]

    next_obs, reward, done, _, _ = env.step(action)

    theta_log.append(obs[0])
    theta_dot_log.append(obs[1])
    target_log.append(obs[2])
    action_log.append(action[0])
    reward_log.append(reward)

    obs = next_obs


np.savez_compressed("lstm_rollout_log.npz",
    theta=np.array(theta_log),
    theta_dot=np.array(theta_dot_log),
    target=np.array(target_log),
    action=np.array(action_log),
    reward=np.array(reward_log)
)


t = np.arange(len(theta_log))

plt.figure()
plt.plot(t, theta_log, label='θ (LSTM)')
plt.plot(t, target_log, label='θ target', linestyle='--')
plt.xlabel('Time step')
plt.ylabel('Angle (rad)')
plt.title('LSTM θ vs Time')
plt.legend()
plt.savefig("images/lstm_theta_vs_time.png")

plt.figure()
plt.plot(t, action_log, label='Torque (LSTM)', color='green')
plt.xlabel('Time step')
plt.ylabel('Torque')
plt.title('LSTM Action vs Time')
plt.legend()
plt.savefig("images/lstm_action_vs_time.png")

plt.figure()
plt.plot(t, reward_log, label='Reward (LSTM)', color='red')
plt.xlabel('Time step')
plt.ylabel('Reward')
plt.title('LSTM Reward vs Time')
plt.legend()
plt.savefig("images/lstm_reward_vs_time.png")

print("✅ LSTM rollout complete. Visuals saved to /images")
