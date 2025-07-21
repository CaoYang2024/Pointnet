import torch
import torch.nn.functional as F
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.nn import MLP, radius_graph
from torch_geometric.data import Data

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, mlp_channels):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.mlp = MLP([mlp_channels[0], mlp_channels[1], mlp_channels[2]])
        self.conv = PointConv(local_nn=self.mlp)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx])
        edge_index = torch.stack([col, row], dim=0)

        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class PointNetPP(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.sa1 = SAModule(0.5, 0.2, [3, 64, 64])
        self.sa2 = SAModule(0.25, 0.4, [64, 128, 128])
        self.mlp = MLP([128, 256, 512])

        self.fc1 = torch.nn.Linear(512, 256)
        self.dropout = torch.nn.Dropout(0.4)
        self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x = data.x if data.x is not None else pos

        x, pos, batch = self.sa1(x, pos, batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x = global_max_pool(x, batch)
        x = self.mlp(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
