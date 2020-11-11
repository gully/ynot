import pytest
import torch
import time
from ynot.datasets import FPADataset
from ynot.echelle import Echellogram
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


@pytest.mark.parametrize(
    "device", ["cuda", "cpu"],
)
def test_forward_backward(device):
    """Do the scene models have the right shape"""
    echellogram = Echellogram(device=device)
    t0 = time.time()
    scene_model = echellogram.forward(1)
    t1 = time.time()
    scalar = scene_model.sum()
    t2 = time.time()
    scalar.backward()
    t3 = time.time()
    net_time = t1 - t0
    net_time2 = t3 - t2
    print(f"\n\t{echellogram.device}: forward {net_time:0.5f} seconds", end="\t")
    print(f"\n\t{echellogram.device}: backward {net_time2:0.5f} seconds", end="\t")
    assert scene_model.shape == echellogram.xx.shape
    assert scene_model.dtype == echellogram.xx.dtype



@pytest.mark.parametrize(
    "device", ["cuda", "cpu"]
)
@pytest.mark.slow
def test_training_loop(device):
    """The end-to-end training should operate"""

    model = Echellogram(device=device)
    dataset = FPADataset()

    n_frames_per_batch=1
    train_loader = DataLoader(dataset=dataset, batch_size=n_frames_per_batch, pin_memory=True,
    shuffle=True)

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), 0.01)

    n_epochs = 10

    losses = []
    initial_params = model.parameters()

    t0 = time.time()
    for epoch in range(n_epochs):
        for data in train_loader:
            ind, y_batch = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
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
