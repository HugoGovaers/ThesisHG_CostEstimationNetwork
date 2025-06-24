import torch

from models.Edge_Encoder import (
    EdgeAttributeEncoderMLP,
    EdgeAttributeEncoderZeros,
    EdgeGridEncoder1D,
    EdgeGridEncoderZeros,
)
from models.Face_Encoder import (
    FaceAttributeEncoderMLP,
    FaceAttributeEncoderZeros,
    FaceGridEncoder2D,
    FaceGridEncoderZeros,
)
from models.Graph_NN import GraphNetworkGENConv
from models.Regression import RegressionModel, RegressionTest


class GENConvRegressionModel(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=32,
        face_attr_dim=14,
        face_attr_emb=32,
        edge_attr_dim=15,
        edge_attr_emb=32,
        face_grid_dim=7,
        face_grid_emb=32,
        edge_grid_dim=12,
        edge_grid_emb=32,
        categorical_features_dim=18,
        numerical_features_dim=3,
        dropout_face_encoder=0.0,
        dropout_edge_encoder=0.0,
        dropout_face_grid_encoder=0.0,
        dropout_edge_grid_encoder=0.0,
        dropout_graph_nn=0.0,
        dropout_regression=0.0,
    ):
        super().__init__()

        self.face_encoder = FaceAttributeEncoderMLP(
            face_attr_dim=face_attr_dim,
            face_attr_emb=face_attr_emb,
            dropout=dropout_face_encoder,
        )

        self.face_grid_encoder = FaceGridEncoder2D(
            face_grid_dim=face_grid_dim,
            face_grid_emb=face_grid_emb,
            dropout=dropout_face_grid_encoder,
        )

        self.edge_encoder = EdgeAttributeEncoderMLP(
            edge_attr_dim=edge_attr_dim,
            edge_attr_emb=edge_attr_emb,
            dropout=dropout_edge_encoder,
        )

        self.edge_grid_encoder = EdgeGridEncoder1D(
            edge_grid_dim=edge_grid_dim,
            edge_grid_emb=edge_grid_emb,
            dropout=dropout_edge_grid_encoder,
        )

        self.graph_gen_conv = GraphNetworkGENConv(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim * 2,
            face_encoder=self.face_encoder,
            edge_encoder=self.edge_encoder,
            face_grid_encoder=self.face_grid_encoder,
            edge_grid_encoder=self.edge_grid_encoder,
            dropout=dropout_graph_nn,
        )

        self.regression_model = RegressionModel(
            graph_model=self.graph_gen_conv,
            input_dim=hidden_dim * 2 + categorical_features_dim + numerical_features_dim,
            dropout=dropout_regression,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        face_grid,
        edge_grid,
        features_categorical,
        features_numerical,
        batch,
        return_activations=False,
    ):
        return self.regression_model(x, edge_index, edge_attr, face_grid, edge_grid, features_categorical, features_numerical, batch, return_activations)

class GENConvRegressionModel_FaceAttributes(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=32,
        face_attr_dim=14,
        face_attr_emb=32,
        edge_attr_dim=15,
        edge_attr_emb=32,
        face_grid_dim=7,
        face_grid_emb=32,
        edge_grid_dim=12,
        edge_grid_emb=32,
        categorical_features_dim=18,
        numerical_features_dim=3,
        dropout_face_encoder=0.0,
        dropout_edge_encoder=0.0,
        dropout_face_grid_encoder=0.0,
        dropout_edge_grid_encoder=0.0,
        dropout_graph_nn=0.0,
        dropout_regression=0.0,
    ):
        super().__init__()

        self.face_encoder = FaceAttributeEncoderMLP(
            face_attr_dim=face_attr_dim,
            face_attr_emb=face_attr_emb,
            dropout=dropout_face_encoder,
        )

        self.face_grid_encoder = FaceGridEncoder2D(
            face_grid_dim=face_grid_dim,
            face_grid_emb=face_grid_emb,
            dropout=dropout_face_grid_encoder,
        )

        self.edge_encoder = EdgeAttributeEncoderZeros(
            edge_attr_dim=edge_attr_dim,
            edge_attr_emb=edge_attr_emb,
            dropout=dropout_edge_encoder,
        )

        self.edge_grid_encoder = EdgeGridEncoderZeros(
            edge_grid_dim=edge_grid_dim,
            edge_grid_emb=edge_grid_emb,
            dropout=dropout_edge_grid_encoder,
        )

        self.graph_gen_conv = GraphNetworkGENConv(
            input_dim=    hidden_dim * 2,
            hidden_dim=hidden_dim * 2,
            face_encoder=self.face_encoder,
                edge_encoder=self.edge_encoder,
            face_grid_encoder=self.face_grid_encoder,
            edge_grid_encoder=self.edge_grid_encoder,
            dropout=dropout_graph_nn,
        )

        self.regression_model = RegressionModel(
            graph_model=self.graph_gen_conv,
            input_dim=hidden_dim * 2
            + categorical_features_dim
            + numerical_features_dim,
            dropout=dropout_regression,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        face_grid,
        edge_grid,
        features_categorical,
        features_numerical,
        batch,
        return_activations=False,
    ):
        return self.regression_model(
            x,
            edge_index,
            edge_attr,
            face_grid,
            edge_grid,
            features_categorical,
            features_numerical,
            batch,
            return_activations,
        )

class RegressionModelTest(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=32,
        categorical_features_dim=18,
        numerical_features_dim=3,
        dropout_regression=0.0,
    ):
        super().__init__()

        self.regression_model = RegressionTest(
            input_dim=categorical_features_dim
            + numerical_features_dim,
            dropout=dropout_regression,
        )

    def forward(
        self,
        features_categorical,
        features_numerical,
        batch,
        return_activations=False,
    ):
        return self.regression_model(
            features_categorical,
            features_numerical,
            batch,
            return_activations,
        )
