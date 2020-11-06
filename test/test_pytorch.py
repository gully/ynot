import torch
from torch import nn


class SimpleNN(nn.Module):

    def __init__(self, bounds=(10, 15)):
        super().__init__()
        self.mm = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.bb = nn.Parameter(torch.tensor([-2.0], requires_grad=True))

        self.y0 = bounds[1]

    def forward(self, x):
        return self.mm * x + self.bb + self.y0


def test_torch():
    """Make sure we know how to cast dypes and devices of models"""
    model = SimpleNN()

    device = 'cuda'

    model = model.to(device)
    x = torch.arange(10.0).to(device)
    output = model.forward(x)

    assert x.device.type == device
    assert model.mm.device.type == device
    assert output.device.type == device
    assert x.dtype == torch.float32
    assert model.mm.dtype == torch.float32
    assert output.dtype == torch.float32

    model = model.double()
    x = x.double()
    output = model.forward(x)

    assert x.device.type == device
    assert model.mm.device.type == device
    assert output.device.type == device
    assert x.dtype == torch.float64
    assert model.mm.dtype == torch.float64
    assert output.dtype == torch.float64
