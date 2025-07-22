# 🌪️ AI-Driven Torque Control for BLDC Motors Using Dreamer and LSTM

This repository contains the full implementation, experiments, and evaluation of a flagship control system that compares two learning-based policies — **Dreamer (model-based reinforcement learning)** and a **supervised LSTM** — for torque control in a simulated motor environment. The project targets real-time embedded deployment (e.g., STM32) and is designed for applications in humanoid robotics, wearable systems, and energy-efficient actuation.

---

## 📘 Paper Summary

The paper titled **"Sim2Real Control with Dreamer: Low-Latency AI Torque Control for Embedded Motor Drivers"** explores:

- Real-time torque control using learned policies
- Phase-space and reward stability comparisons between Dreamer and LSTM
- Deployment-ready ONNX actor exports with sub-millisecond inference
- Full rollout diagnostics under dynamic target switching

📄 **[LaTeX Paper](./paper/main.tex)** included.

---


## 🛠️ Project Structure

```bash
MotorDriverSoftware/
├── collect_motor_dataset.py         # Expert data generator
├── train_dreamer.py                 # Dreamer training script
├── train_lstm.py                    # LSTM training script
├── dreamer_model.py                 # Dreamer model architecture
├── lstm_policy.py                   # LSTM model definition
├── evaluate_actor_policy.py         # Dreamer policy rollout
├── evaluate_lstm_policy.py          # LSTM policy rollout
├── visualize_rollout.py             # Basic rollout visualization
├── analyze_rollout.py               # Tracking error + reward analysis
├── export_actor_to_onnx.py          # Dreamer ONNX export
├── benchmark_onnx_latency.py        # Latency test for Dreamer actor
├── images/                          # All generated graphs and plots
│   ├── tracking_performance_dreamer_lstm.png
│   ├── final_reward_and_error_comparison.png
│   ├── phase_space_dreamer_lstm.png
│   └── ... (more)
└── actor_rollout_log.npz            # Dreamer rollout (θ, θ̇, τ, r)
└── lstm_rollout_log.npz             # LSTM rollout (θ, θ̇, τ, r)

```

1. Clone this repository
```bash
git clone https://github.com/yourusername/ai-motor-control.git
cd ai-motor-control
```

2. Setup Python Environment (Conda Recommended)
```bash
conda create -n ppo_motor python=3.10
conda activate ppo_motor
pip install -r requirements.txt
```

3. Train Dreamer or LSTM
```bash
python train_dreamer.py  # or
python train_lstm.py
```

4. Evaluate and Visualize Rollouts
```bash
python evaluate_actor_policy.py
python visualize_rollout.py
```
5. Export to ONNX and Benchmark Latency
```bash
python export_actor_to_onnx.py
python benchmark_onnx_latency.py
```

## 🤖 Highlights

✅ Real-time deployment capability (Dreamer actor < 0.12 ms)

✅ Robust against dynamic targets and disturbances

✅ Modular architecture with RSSM, decoder, reward, actor

✅ ONNX export for STM32-ready deployment

✅ Full paper and plots included for reproducibility

#  Contact
Questions or collaborations? Reach out:

📧 jaival[dot]email[at]utoronto.ca

🧠 Portfolio

