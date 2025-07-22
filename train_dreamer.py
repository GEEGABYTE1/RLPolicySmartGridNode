import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dreamer_model import (
    Encoder, RSSM, Decoder, RewardPredictorActor,
    RewardPredictorCritic, Actor, Critic
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# --- Hyperparameters ---
obs_dim = 3
action_dim = 1
latent_dim = 16
batch_size = 64
epochs = 600
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = np.load("motor_dataset_normalized.npz")
obs = torch.tensor(data["obs"], dtype=torch.float32)
action = torch.tensor(data["action"], dtype=torch.float32)
next_obs = torch.tensor(data["next_obs"], dtype=torch.float32)
reward = torch.tensor(data["reward"], dtype=torch.float32).unsqueeze(-1)

dataset = TensorDataset(obs, action, next_obs, reward)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


encoder = Encoder(obs_dim, latent_dim).to(device)
rssm = RSSM(latent_dim, action_dim).to(device)
decoder = Decoder(latent_dim, obs_dim).to(device)
reward_predictor_actor = RewardPredictorActor(latent_dim).to(device)
reward_predictor_critic = RewardPredictorCritic(latent_dim).to(device)
actor = Actor(latent_dim, action_dim).to(device)
critic = Critic(latent_dim).to(device)


model_params = (
    list(encoder.parameters())
    + list(rssm.parameters())
    + list(decoder.parameters())
    + list(reward_predictor_actor.parameters())
    + list(reward_predictor_critic.parameters())
)
model_opt = optim.Adam(model_params, lr=1e-3)
actor_opt = optim.Adam(actor.parameters(), lr=1e-3)
critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

for epoch in range(epochs):
    total_loss = 0
    for obs_batch, act_batch, next_obs_batch, reward_batch in tqdm(loader):
        obs_batch = obs_batch.to(device)
        act_batch = act_batch.to(device)
        next_obs_batch = next_obs_batch.to(device)
        reward_batch = reward_batch.to(device)


        z = encoder(obs_batch)
        h = torch.zeros(1, z.size(0), 128).to(device)
        z_next, h = rssm(z, act_batch, h)

        obs_recon = decoder(z_next)
        reward_pred = reward_predictor_critic(z_next)  

        recon_loss = F.mse_loss(obs_recon, next_obs_batch)
        reward_loss = F.mse_loss(reward_pred, reward_batch)

        model_opt.zero_grad()
        (recon_loss + reward_loss).backward()
        model_opt.step()

   
        z_actor = encoder(obs_batch.detach())
        imagined_reward_actor = reward_predictor_actor(z_actor)
        actor_loss = (-1.0) * imagined_reward_actor.mean()

        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        
        z_critic = encoder(obs_batch.detach())
        imagined_reward_critic = reward_predictor_critic(z_critic).detach()
        value_pred = critic(z_critic)
        critic_loss = F.mse_loss(value_pred, imagined_reward_critic)

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        total_loss += recon_loss.item() + reward_loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss:.4f}")

torch.save(actor.state_dict(), "actor.pt")
print("✅ Saved trained actor to actor.pt")