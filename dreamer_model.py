import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, obs):
        return self.fc(obs)


# RSSM: (z_t, a_t) -> z_{t+1}
class RSSM(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.rnn = nn.GRU(latent_dim + action_dim, hidden_dim, batch_first=True)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_prev, a_prev, h_prev):
        x = torch.cat([z_prev, a_prev], dim=-1).unsqueeze(1)  
        out, h = self.rnn(x, h_prev)                           
        z_next = self.latent_proj(out.squeeze(1))             
        return z_next, h



class Decoder(nn.Module):
    def __init__(self, latent_dim, obs_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim)
        )

    def forward(self, z):
        return self.fc(z)


class RewardPredictorActor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        return self.fc(z)



class RewardPredictorCritic(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        return self.fc(z)



class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.fc(z)



class Critic(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        return self.fc(z)
