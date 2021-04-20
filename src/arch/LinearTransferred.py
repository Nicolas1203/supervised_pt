import torch


class LinearTransferred(torch.nn.Module):
    def __init__(self, backbone, low_dim,  out_dim):
        super(LinearTransferred, self).__init__()
        self.backbone = backbone
        self.low_dim = low_dim
        self.out_dim = out_dim
        self.fc_out = torch.nn.Linear(self.low_dim, self.out_dim)

    def forward(self, x):
        return self.fc_out(self.backbone(x))
    