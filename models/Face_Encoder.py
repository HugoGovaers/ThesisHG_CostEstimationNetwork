import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceAttributeEncoderMLP(torch.nn.Module):
    """
    Initialize the face attribute encoder.

    Parameters:
    - face_attr_dim: Number of face attributes.
    - face_attr_emb: Output size (face attribute embeddings).
    - hidden_dim: Number of hidden nodes in the first fully connected layer.
    - dropout: Dropout rate for the fully connected layers.
    """
    def __init__(self, face_attr_dim=14, face_attr_emb=32, hidden_dim=32, dropout=0.5):
        super(FaceAttributeEncoderMLP, self).__init__()
        self.fc1 = torch.nn.Linear(face_attr_dim, hidden_dim)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)  # Batch normalization layer

        self.fc2 = torch.nn.Linear(hidden_dim, face_attr_emb)
        self.norm2 = torch.nn.LayerNorm(face_attr_emb)  # Batch normalization layer

        self.act = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, return_activations=False):
        x = self.fc1(x)
        x = self.norm1(x)
        act1 = self.act(x)
        x = self.dropout(act1)

        x = self.fc2(x)
        x = self.norm2(x)
        act2 = self.act(x)
        x = self.dropout(act2)

        if return_activations:
            face_attr_activations = {
                "face_act1": act1,
                "face_act2": act2,
                }
            return x, face_attr_activations

        return x


class FaceGridEncoder2D(nn.Module):
    """
    Initialize the face grid encoder.

    Parameters:
    - face_grid_dim: Number of face grid attributes.
    - face_grid_emb: Output size (face grid embeddings).
    - dropout: Dropout rate for the fully connected layers.
    """
    def __init__(self, face_grid_dim=7, face_grid_emb=32, dropout=0.0):
        super(FaceGridEncoder2D, self).__init__()
        self.face_grid_dim = face_grid_dim
        self.conv1 = nn.Conv2d(face_grid_dim, 16, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, face_grid_emb)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, face_grid, return_activations=False):
        # face_grid: [num_faces, channels, height, width]
        assert face_grid.size(1) == self.face_grid_dim
        batch_size = face_grid.size(0)

        x = self.conv1(face_grid)
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

        x = self.pool(x)  # shape: [num_faces, 64]
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.dropout(x)

        if return_activations:
            activations = {
                "face_conv1_act": act1,
                "face_conv2_act": act2,
                "face_act": act3,
                "face_fc_output": x,
            }
            return x, activations

        return x

class FaceAttributeEncoderZeros(nn.Module):
    """
    Custom face attribute encoder that outputs zeros for testings
    """
    def __init__(self, face_attr_dim, face_attr_emb, **kwargs):
        super(FaceAttributeEncoderZeros, self).__init__()
        self.face_attr_emb = face_attr_emb

    def forward(self, face_attr, return_activations=False):
        batch_size = face_attr.size(0)
        zeros = torch.zeros(
            batch_size,
            self.face_attr_emb,
            device=face_attr.device,
            dtype=face_attr.dtype,
        )
        if return_activations:
            return zeros, {}
        return zeros


class FaceGridEncoderZeros(nn.Module):
    """
    Custom face grid encoder that outputs zeros for testings
    """
    def __init__(self, face_grid_dim, face_grid_emb, **kwargs):
        super(FaceGridEncoderZeros, self).__init__()
        self.face_grid_emb = face_grid_emb

    def forward(self, face_grid, return_activations=False):
        batch_size = face_grid.size(0)
        zeros = torch.zeros(
            batch_size,
            self.face_grid_emb,
            device=face_grid.device,
            dtype=face_grid.dtype,
        )
        if return_activations:
            return zeros, {}
        return zeros
