import pytest
import torch
import time
from ynot.datasets import FPADataset
from ynot.echelle import Echellogram
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

@pytest.mark.parametrize(
    "device", ["cuda"]
)
@pytest.mark.slow
def test_training_loop(device):
    """The end-to-end training should operate"""

    model = Echellogram(device=device)
    dataset = FPADataset()

    n_frames_per_batch=1
    train_loader = DataLoader(dataset=dataset, batch_size=n_frames_per_batch, shuffle=True)

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), 0.01)

    n_epochs = 10

    losses = []
    initial_params = model.parameters()

    t0 = time.time()
    for epoch in range(n_epochs):
        for data in train_loader:
            ind, y_batch = data[0].to(device), data[1].to(device)
            model.train()
            yhat = model.forward(ind).unsqueeze(0)
            loss = loss_fn(yhat, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
    t1 = time.time()
    net_time = t1 - t0
    print(f"\n\t {n_epochs} epochs on {device}: {net_time:0.1f} seconds", end="\t")

    for loss in losses:
        assert loss == loss

    for parameter in model.parameters():
        assert parameter.isfinite().all()
