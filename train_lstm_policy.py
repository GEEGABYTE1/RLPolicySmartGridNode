import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from lstm_policy import LSTMPolicy


input_dim = 3
seq_len = 5
hidden_dim = 64
batch_size = 64
epochs = 10
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = np.load("motor_dataset_normalized.npz")
obs = data["obs"]
action = data["action"]
print("action min:", np.min(action), "max:", np.max(action))

X_seq = []
y_seq = []
for i in range(len(obs) - seq_len):
    X_seq.append(obs[i:i+seq_len])
    y_seq.append(action[i+seq_len-1])  

X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32)

dataset = TensorDataset(X_seq, y_seq)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = LSTMPolicy(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()


for epoch in range(epochs):
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


torch.save(model.state_dict(), "lstm_actor.pt")
print("✅ Saved trained LSTM actor to lstm_actor.pt")
