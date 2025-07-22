import torch
import torch.nn as nn

class LSTMPolicy(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Tanh()  
        )

    def forward(self, x, hidden=None):
        """
        x: Tensor of shape [batch_size, seq_len, input_dim]
        hidden: (h_0, c_0), optional initial hidden state

        Returns:
        - out: Tensor of shape [batch_size, output_dim]
        """
        lstm_out, hidden = self.lstm(x, hidden)
        out = self.fc(lstm_out[:, -1])  
        return out
