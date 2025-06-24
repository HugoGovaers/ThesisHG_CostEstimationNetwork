import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAttributeEncoderMLP(torch.nn.Module):
    """
    Initialize the edge attribute encoder.

    Parameters:
    - edge_attr_dim: Number of edge attributes.
    - edge_attr_emb: Output size (edge attribute embeddings).
    - hidden_dim: Number of hidden nodes in the first fully connected layer.
    - dropout: Dropout rate for the fully connected layers.
    """
    def __init__(
        self, edge_attr_dim, edge_attr_emb, hidden_dim=32, dropout=0.5
    ):
        super(EdgeAttributeEncoderMLP, self).__init__()
        self.fc1 = torch.nn.Linear(edge_attr_dim, hidden_dim)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)

        self.fc2 = torch.nn.Linear(hidden_dim, edge_attr_emb)
        self.norm2 = torch.nn.LayerNorm(edge_attr_emb)

        self.act = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, edge_attr, return_activations=False):
        x = self.fc1(edge_attr)
        x = self.norm1(x)
        act1 = self.act(x)
        x = self.dropout(act1)

        x = self.fc2(x)
        x = self.norm2(x)
        act2 = self.act(x)
        x = self.dropout(act2)

        if return_activations:
            face_attr_activations = {
                "edge_act1": act1,
                "edge_act2": act1,
            }
            print(f"face_attr_activations: {face_attr_activations}")
            return x, face_attr_activations

        return x

class EdgeGridEncoder1D(nn.Module):
    """
    Initialize the edge grid encoder.

    Parameters:
    - edge_grid_dim: Number of edge grid attributes.
    - edge_grid_emb: Output size (edge grid embeddings).
    - dropout: Dropout rate for the fully connected layers.
    """
    def __init__(self, edge_grid_dim, edge_grid_emb, dropout=0.0):
        super(EdgeGridEncoder1D, self).__init__()
        self.edge_grid_dim = edge_grid_dim
        self.conv1 = nn.Conv1d(edge_grid_dim, 16, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, edge_grid_emb)
        self.dropout = nn.Dropout(dropout)

        self.act = torch.nn.GELU()

    def forward(self, edge_grid, return_activations=False):
        # edge_grid: [num_edges, channels, grid_size]
        assert edge_grid.size(1) == self.edge_grid_dim
        batch_size = edge_grid.size(0)

        x = self.conv1(edge_grid)
        x = self.norm1(x)
        act1 = self.act(x)
        x = self.dropout(act1)

        x = self.conv2(x)
        x = self.norm2(x)
        act2 = self.act(x)
        x = self.dropout(act2)

        x = self.conv3(x)
        x = self.norm3(x)
        act3 = self.act(x)
        x = self.dropout(act3)

        x = self.pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.dropout(x)

        if return_activations:
            activations = {
            "edge_conv1_act": act1,
            "edge_conv2_act": act2,
            "edge_conv3_act": act3,
            "edge_fc_output": x,
            }
            return x, activations

        return x

class EdgeAttributeEncoderZeros(nn.Module):
    """
    Custom edge attribute encoder that outputs zeros for testings
    """
    def __init__(self, edge_attr_dim, edge_attr_emb, **kwargs):
        super(EdgeAttributeEncoderZeros, self).__init__()
        self.edge_attr_emb = edge_attr_emb

    def forward(self, edge_attr, return_activations=False):
        batch_size = edge_attr.size(0)
        zeros = torch.zeros(batch_size, self.edge_attr_emb, device=edge_attr.device, dtype=edge_attr.dtype)
        if return_activations:
            return zeros, {}
        return zeros

class EdgeGridEncoderZeros(nn.Module):
    """
    Custom edge grid encoder that outputs zeros for testings
    """
    def __init__(self, edge_grid_dim, edge_grid_emb, **kwargs):
        super(EdgeGridEncoderZeros, self).__init__()
        self.edge_grid_emb = edge_grid_emb

    def forward(self, edge_grid, return_activations=False):
        batch_size = edge_grid.size(0)
        zeros = torch.zeros(batch_size, self.edge_grid_emb, device=edge_grid.device, dtype=edge_grid.dtype)
        if return_activations:
            return zeros, {}
        return zeros
