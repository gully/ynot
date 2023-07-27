import torch
import time
from ynot.datasets import FPADataset, IGRINSDataset
from ynot.igrins import IGRINSEchellogram
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import argparse
import webbrowser

parser = argparse.ArgumentParser(
    description="Experimental astronomical echellogram inference"
)
parser.add_argument(
    "--resume", action="store_true", help="Resume model from last existing saved model",
)
parser.add_argument(
    "--n_epochs", default=1800, type=int, help="Number of training epochs"
)

args = parser.parse_args()

writer = SummaryWriter(log_dir="runs/exp2")


def plot_scene_model(images):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch
    """
    fig, axes = plt.subplots(3, figsize=(8, 3))
    axes[0].imshow(images[0].numpy().T, vmin=0, vmax=2500, origin="lower")
    axes[1].imshow(images[1].numpy().T, vmin=0, vmax=2500, origin="lower")
    axes[2].imshow(
        images[0].numpy().T - images[1].numpy().T, vmin=-100, vmax=100, origin="lower"
    )
    return fig


# Change to 'cpu' if you do not have an NVIDIA GPU
# Warning, it will be about 30X slower.
device = "cuda"

model = IGRINSEchellogram(device=device, dense_sky=True, ybounds=(490, 510))
model = model.to(device, non_blocking=True)
dataset = IGRINSDataset(
    root_dir="../test/data/GS-2020B-Q-318/20201202/",
    ybounds=(490, 510),
    inpaint_bad_pixels=False,
    inpaint_cosmic_rays=False,
)

# Initialize from a previous training run
if args.resume:
    # for key in model.state_dict():
    #    model.state_dict()[key] *=0
    #    model.state_dict()[key] += state_dict[key].to(device)
    state_dict = torch.load("model_coeffs.pt")
    model.load_state_dict(state_dict)

# Only send one frame per batch
n_frames_per_batch = 1
train_loader = DataLoader(dataset=dataset, batch_size=n_frames_per_batch, shuffle=True)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), 0.004)

# It currently takes 0.5 seconds per training epoch, for about 7200 epochs per hour
webbrowser.open("http://localhost:6006/", new=2)
n_epochs = args.n_epochs

losses = []

# Mask the last 24 pixel columns
mask = model.xx < 2048

t0 = time.time()
t_iter = trange(n_epochs, desc="Training", leave=True)
for epoch in t_iter:
    for i, data in enumerate(train_loader, 0):
        ind, y_batch = (
            data[0].to(device, non_blocking=True),
            data[1].to(device, non_blocking=True),
        )
        model.train()
        y_batch_vec = y_batch.squeeze()[mask]
        yhat = model.forward(ind)
        yhat_vec = yhat[mask]
        loss = loss_fn(yhat, y_batch)
        loss.backward(retain_graph=False)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    writer.add_scalar("loss", loss.item(), global_step=epoch * len(train_loader) + i)
    t_iter.set_description(f"Loss {loss.item(): 15.3f}")
    t_iter.refresh()
    if (epoch % 60) == 0:
        torch.save(model.state_dict(), "model_coeffs.pt")
        writer.add_figure(
            "predictions vs. actuals",
            plot_scene_model([y_batch.squeeze().cpu(), yhat.detach().squeeze().cpu()]),
            global_step=epoch * len(train_loader) + i,
        )

# Save the model parameters for next time
t1 = time.time()
net_time = t1 - t0
print(f"{n_epochs} epochs on {device}: {net_time:0.1f} seconds", end="\t")
torch.save(model.state_dict(), "model_coeffs.pt")
writer.close()
