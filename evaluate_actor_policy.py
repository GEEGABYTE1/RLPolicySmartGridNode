import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from motor_env_dreamer import MotorEnvDreamer
from dreamer_model import Encoder, Actor


os.makedirs("images", exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs_dim = 3
latent_dim = 16
action_dim = 1

encoder = Encoder(obs_dim, latent_dim).to(device)
actor = Actor(latent_dim, action_dim).to(device)
actor.load_state_dict(torch.load("actor.pt", map_location=device))

encoder.eval()
actor.eval()

env = MotorEnvDreamer(render=False)
obs, _ = env.reset()

theta_log = []
theta_dot_log = []
target_log = []
action_log = []
reward_log = []


for t in range(1000):
    norm_obs = np.array([
        obs[0] / np.pi,
        obs[1] / 10.0,
        obs[2] / np.pi
    ], dtype=np.float32)

    obs_tensor = torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        z = encoder(obs_tensor)
        action = actor(z).cpu().numpy()[0]

    next_obs, reward, done, _, _ = env.step(action)

    theta_log.append(obs[0])
    theta_dot_log.append(obs[1])
    target_log.append(obs[2])
    action_log.append(action[0])
    reward_log.append(reward)

    obs = next_obs


np.savez_compressed("actor_rollout_log.npz",
    theta=np.array(theta_log),
    theta_dot=np.array(theta_dot_log),
    target=np.array(target_log),
    action=np.array(action_log),
    reward=np.array(reward_log)
)


def pid_controller(obs, target, kp=10.0, kd=0.5):
    theta, theta_dot = obs[0], obs[1]
    error = target - theta
    derivative = -theta_dot
    return np.clip(kp * error + kd * derivative, -2.0, 2.0)


obs, _ = env.reset()

theta_pid = []
theta_dot_pid = []
action_pid = []
reward_pid = []

for t in range(1000):
    target = obs[2]
    torque = pid_controller(obs, target)

    # Inject disturbance at t = 500
    if t == 500:
        torque += 1.0
        print("⚠️  Disturbance injected at t = 500")

    next_obs, reward, done, _, _ = env.step([torque])

    theta_pid.append(obs[0])
    theta_dot_pid.append(obs[1])
    action_pid.append(torque)
    reward_pid.append(reward)

    obs = next_obs


t = np.arange(len(theta_log))

plt.figure()
plt.plot(t, theta_log, label='Dreamer θ')
plt.plot(t, theta_pid, label='PID θ', linestyle='--')
plt.plot(t, target_log, label='θ target', linestyle=':')
plt.xlabel('Time step')
plt.ylabel('Angle (rad)')
plt.legend()
plt.title('θ vs Time (Dreamer vs PID)')
plt.savefig("images/theta_comparison.png")

plt.figure()
plt.plot(t, action_log, label='Dreamer Torque')
plt.plot(t, action_pid, label='PID Torque', linestyle='--')
plt.xlabel('Time step')
plt.ylabel('Torque')
plt.legend()
plt.title('Torque Comparison')
plt.savefig("images/torque_comparison.png")

plt.figure()
plt.plot(t, reward_log, label='Dreamer Reward')
plt.plot(t, reward_pid, label='PID Reward', linestyle='--')
plt.xlabel('Time step')
plt.ylabel('Reward')
plt.legend()
plt.title('Reward Comparison')
plt.savefig("images/reward_comparison.png")

plt.figure()
plt.plot(theta_log, theta_dot_log)
plt.xlabel('θ (rad)')
plt.ylabel('θ̇ (rad/s)')
plt.title('Phase Plot (Dreamer Policy)')
plt.grid(True)
plt.savefig("images/phase_plot.png")

print("✅ Evaluation complete. Plots saved in /images")
