import numpy as np
import pandas as pd


dreamer_data = np.load("actor_rollout_log.npz")
lstm_data = np.load("lstm_rollout_log.npz")

theta_dreamer = dreamer_data["theta"]
target_dreamer = dreamer_data["target"]
reward_dreamer = dreamer_data["reward"]

theta_lstm = lstm_data["theta"]
target_lstm = lstm_data["target"]
reward_lstm = lstm_data["reward"]


def get_convergence_step(theta_seq, target_seq, tol=0.05):
    error = np.abs(theta_seq - target_seq)
    for i in range(len(error)):
        if np.all(error[i:i+50] < tol):
            return i
    return -1

def get_overshoot(theta_seq, target_seq):
    target_val = target_seq[-1]
    peak = np.max(np.abs(theta_seq - target_val))
    return peak

def reward_stats(r):
    return np.mean(r), np.min(r), np.max(r)

def get_recovery_time(theta_seq, target_seq, t_disturb=500, tol=0.05):
    error = np.abs(theta_seq[t_disturb:] - target_seq[t_disturb:])
    for i in range(len(error)):
        if np.all(error[i:i+30] < tol):
            return i
    return -1


metrics = {
    "Method": ["Dreamer", "LSTM"],
    "Convergence Step": [
        get_convergence_step(theta_dreamer, target_dreamer),
        get_convergence_step(theta_lstm, target_lstm)
    ],
    "Max Overshoot (rad)": [
        get_overshoot(theta_dreamer, target_dreamer),
        get_overshoot(theta_lstm, target_lstm)
    ],
    "Avg Reward": [
        reward_stats(reward_dreamer)[0],
        reward_stats(reward_lstm)[0]
    ],
    "Reward Min": [
        reward_stats(reward_dreamer)[1],
        reward_stats(reward_lstm)[1]
    ],
    "Reward Max": [
        reward_stats(reward_dreamer)[2],
        reward_stats(reward_lstm)[2]
    ],
    "Recovery Steps (from t=500)": [
        get_recovery_time(theta_dreamer, target_dreamer),
        get_recovery_time(theta_lstm, target_lstm)
    ]
}

df = pd.DataFrame(metrics)
df.to_csv("rollout_metrics_dreamer_lstm.csv", index=False)
print("✅ Saved Dreamer vs LSTM metrics to rollout_metrics_dreamer_lstm.csv")
print(df)
