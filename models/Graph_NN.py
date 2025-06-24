import torch
import torch.nn.functional as F
from torch_geometric.nn import GENConv


class GraphNetworkGENConv(torch.nn.Module):
    """
    Initialize the graph network with GENConv layers.
    https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.nn.conv.GENConv.html.
    This network uses multiple GENConv layers with skip connections and dropout for regularization.
    GENConv is a generalized edge convolution layer that can handle edge attributes and supports various aggregation methods.
    The network is designed to process graph data with both face and edge attributes, as well as grid encodings for faces and edges.

    Parameters:
    - input_dim: Number of input features (concatenated face and edge attributes).
    - edge_encoder: Encoder model for edge attributes.
    - edge_grid_encoder: Encoder model for edge grid attributes.
    - face_encoder: Encoder model for face attributes.
    - face_grid_encoder: Encoder model for face grid attributes.
    - hidden_dim: Number of hidden nodes.
    - dropout: Dropout rate for the fully connected layers.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        edge_encoder,
        face_encoder,
        face_grid_encoder,
        edge_grid_encoder,
        dropout=0.0,
    ):
        super(GraphNetworkGENConv, self).__init__()
        self.face_encoder = face_encoder
        self.edge_encoder = edge_encoder

        self.edge_grid_encoder = edge_grid_encoder
        self.face_grid_encoder = face_grid_encoder

        # First NNConv layer
        self.conv1 = GENConv(input_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')

        # Second NNConv layer
        self.conv2 = GENConv(input_dim, hidden_dim, aggr="softmax",
                             t=1.0, learn_t=True, num_layers=2, norm="layer",)

        # Third NNConv layer
        self.conv3 = GENConv(input_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')

        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_attr, face_grid, edge_grid, return_activations=False):
        # Pass through the FaceAttributeEncoder

        if return_activations:
            face_attribute_encoded, face_attr_activations = self.face_encoder(x, return_activations=True)
            face_grid_encoded, face_grid_activations = self.face_grid_encoder(face_grid, return_activations=True)
            x_encoded = torch.cat([face_attribute_encoded, face_grid_encoded], dim=-1)

            edge_attr_encoded, edge_attr_activations = self.edge_encoder(edge_attr, return_activations=True)
            edge_grid_encoded, edge_grid_activations = self.edge_grid_encoder(edge_grid, return_activations=True)
            edge_attr_encoded = torch.cat([edge_attr_encoded, edge_grid_encoded], dim=-1)
        else:
            face_attribute_encoded = self.face_encoder(x)
            face_grid_encoded = self.face_grid_encoder(face_grid)
            x_encoded = torch.cat([face_attribute_encoded, face_grid_encoded], dim=-1)

            edge_attr_encoded = self.edge_encoder(edge_attr)
            edge_grid_encoded = self.edge_grid_encoder(edge_grid)
            edge_encoded = torch.cat([edge_attr_encoded, edge_grid_encoded], dim=-1)

        # First NNConv layer
        x1 = self.conv1(x_encoded, edge_index, edge_encoded)
        act1 = self.act(x1)
        x1 = self.dropout(act1)

        # Second NNConv layer with skip connection
        x2 = self.conv2(x1, edge_index, edge_encoded)
        act2 = self.act(x2 + x1)  # Skip connection
        x2 = self.dropout(act2)

        # Third NNConv layer with skip connection
        x3 = self.conv3(x2, edge_index, edge_encoded)
        act3 = self.act(x3 + x2 + x1)  # Skip connection
        x3 = self.dropout(act3)

        if return_activations:
            nnconv_activations = {
                "nnconv_act1": act1,
                "nnconv_act2": act2,
                "nnconv_act3": act3,
            }
            return x3, {**face_attr_activations, **face_grid_activations, **edge_attr_activations, **edge_grid_activations, **nnconv_activations}

        return x3
