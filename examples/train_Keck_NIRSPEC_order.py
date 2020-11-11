import torch
import time
from ynot.datasets import FPADataset
from ynot.echelle import Echellogram
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm, trange

# Change to 'cpu' if you do not have an NVIDIA GPU
# Warning, it will be about 30X slower.
device = 'cuda'

model = Echellogram(device=device)
model = model.to(device, non_blocking=True)
dataset = FPADataset()

# Initialize from a previous training run
state_dict = torch.load('model_coeffs.pt')
#for key in model.state_dict():
#    model.state_dict()[key] *=0
#    model.state_dict()[key] += state_dict[key].to(device)
model.load_state_dict(state_dict)

# Only send one frame per batch
n_frames_per_batch=1
train_loader = DataLoader(dataset=dataset, batch_size=n_frames_per_batch, shuffle=True)

loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), 0.02)

# It currently takes 0.5 seconds per training epoch, for about 7200 epochs per hour
n_epochs = 10

losses = []

t0 = time.time()
t_iter = trange(n_epochs, desc='Training', leave=True)
for epoch in t_iter:
    for data in train_loader:
        ind, y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
        model.train()
        yhat = model.forward(ind).unsqueeze(0)
        loss = loss_fn(yhat, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    t_iter.set_description(f"Loss {loss: 15.3f}")
    t_iter.refresh()
    if ((epoch % 100) == 0) :
        torch.save(model.state_dict(), 'model_coeffs.pt')

# Save the model parameters for next time
t1 = time.time()
net_time = t1 - t0
print(f"{n_epochs} epochs on {device}: {net_time:0.1f} seconds", end="\t")
torch.save(model.state_dict(), 'model_coeffs.pt')
