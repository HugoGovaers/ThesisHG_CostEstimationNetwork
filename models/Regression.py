import torch
from torch_geometric.nn import global_mean_pool


class RegressionModel(torch.nn.Module):
    def __init__(
        self,
        graph_model,
        input_dim,
        dropout=0.0,
    ):
        """
        Initialize the regression model with a graph model.

        Parameters:
        - graph_model: The graph model whose output serves as input to this regression model.
        - input_dim: Number of input features (output of the graph model).
        - dropout: Dropout rate for the fully connected layers.
        """
        super(RegressionModel, self).__init__()
        self.graph_model = graph_model

        self.fc1 = torch.nn.Linear(input_dim, 32)
        self.norm1 = torch.nn.BatchNorm1d(32)  # Batch normalization layer

        self.fc2 = torch.nn.Linear(32, 16)
        self.norm2 = torch.nn.BatchNorm1d(16)  # Batch normalization layer

        self.fc3 = torch.nn.Linear(16, 8)
        self.norm3 = torch.nn.BatchNorm1d(8)  # Batch normalization layer

        self.fc4 = torch.nn.Linear(8, 1)

        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        face_grid,
        edge_grid,
        categorical_features,
        numerical_features,
        batch,
        return_activations=False,
    ):
        if return_activations:
            x, activations = self.graph_model(x, edge_index, edge_attr, face_grid, edge_grid, return_activations=True)
        else:
            x = self.graph_model(x, edge_index, edge_attr, face_grid, edge_grid)

        x = global_mean_pool(x, batch)

        x = torch.cat([x, categorical_features, numerical_features], dim=1)

        x = self.fc1(x)
        x = self.norm1(x)
        act1 = self.act(x)
        x = self.dropout(act1)

        x = self.fc2(x)
        x = self.norm2(x)
        act2 = self.act(x)
        x = self.dropout(act2)

        x = self.fc3(x)
        x = self.norm3(x)
        act3 = self.act(x)
        x = self.dropout(act3)

        x = self.fc4(x)
        output = self.act(x)

        if return_activations:
            regression_activations = {
                "regression_act1": act1,
                "regression_act2": act2,
                "regression_act3": act3,
                "regression_output": output,
            }
            return output, {**activations, **regression_activations}

        return output

class RegressionTest(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        dropout=0.0,
    ):
        """
        Initialize the regression model without a graph model to compare results.

        Parameters:
        - input_dim: Number of input features (output of the graph model).
        - dropout: Dropout rate for the fully connected layers.
        """
        super(RegressionTest, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim, 32)
        self.norm1 = torch.nn.BatchNorm1d(32)

        self.fc2 = torch.nn.Linear(32, 16)
        self.norm2 = torch.nn.BatchNorm1d(16)
        self.fc3 = torch.nn.Linear(16, 8)
        self.norm3 = torch.nn.BatchNorm1d(8)

        self.fc4 = torch.nn.Linear(8, 1)

        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
        self,
        categorical_features,
        numerical_features,
        batch,
        return_activations=False,
    ):
        x = torch.cat([categorical_features, numerical_features], dim=1)

        x = self.fc1(x)
        x = self.norm1(x)
        act1 = self.act(x)
        x = self.dropout(act1)

        x = self.fc2(x)
        x = self.norm2(x)
        act2 = self.act(x)
        x = self.dropout(act2)

        x = self.fc3(x)
        x = self.norm3(x)
        act3 = self.act(x)
        x = self.dropout(act3)

        x = self.fc4(x)
        output = self.act(x)

        if return_activations:
            regression_activations = {
                "regression_act1": act1,
                "regression_act2": act2,
                "regression_act3": act3,
                "regression_output": output,
            }
            return output, {**regression_activations}

        return output
