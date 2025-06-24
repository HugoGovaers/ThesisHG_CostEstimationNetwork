import json
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset


class StepDataset(Dataset):
    def __init__(
        self,
        root,
        reinitialize=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        labels=[],
        calculated=[],
        numerical_features=[],
        categorical_features=[],
    ):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.labels = labels
        self.calculated = calculated
        self.reinitialize = reinitialize
        super(StepDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        # List of raw files to be downloaded.
        return [f for f in os.listdir(self.raw_dir) if f.endswith(".json")]

    @property
    def processed_file_names(self):
        processed_dir = os.path.join(self.root, "processed")
        if self.reinitialize:
            return "reinitialize"
        else:
            return [f for f in os.listdir(processed_dir) if f.startswith("data_")]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _load_product_features(self, file_path):
        with open(file_path, "r") as fp:
            data = json.load(fp)
            product_features = list(data.values())[0]
            product_features_cat = []
            product_features_num = []
            if isinstance(product_features, dict):
                for key, value in product_features.items():
                    if any(key.startswith(cat_feat) for cat_feat in self.categorical_features):
                        product_features_cat.append(value)
                    elif any(key.startswith(num_feat) for num_feat in self.numerical_features):
                        product_features_num.append(value)
            return product_features_cat, product_features_num

    def _load_label(self, file_path, desired_keys=None):
        with open(file_path, "r") as fp:
            data = json.load(fp)
            labels_dict = list(data.values())[0]
            labels = [
                labels_dict[key]
                for key in desired_keys
                if key in labels_dict and isinstance(labels_dict[key], (int, float))
            ]
            return labels

    def _load_one_graph(self, file_path):
        with open(file_path, "r") as fp:
            data = json.load(fp)
            graph_name = data[0]
            attribute_map = data[1]
            graph = attribute_map["graph"]
            graph_edges = graph["edges"]
            face_attr = attribute_map["graph_face_attr"]
            face_grid = attribute_map["graph_face_grid"]
            edge_attr = attribute_map["graph_edge_attr"]
            edge_grid = attribute_map["graph_edge_grid"]

        return (
            graph_name,
            graph,
            graph_edges,
            face_attr,
            face_grid,
            edge_attr,
            edge_grid,
        )

    def process(self):
        desired_keys_labels = self.labels
        desired_keys_calculated = self.calculated
        idx = 0
        for raw_path in self.raw_paths:
            # Read graphs from `raw_path`.
            (
                graph_name,
                graph,
                graph_edges,
                face_attr,
                face_grid,
                edge_attr,
                edge_grid,
            ) = self._load_one_graph(raw_path)
            feature_path = os.path.join(self.root, "features", f"{graph_name}.json")
            label_path = os.path.join(self.root, "labels", f"{graph_name}.json")

            if os.path.exists(feature_path):
                product_features_cat, product_features_num = (
                    self._load_product_features(feature_path)
                )
            else:
                print(f"Feature file for {graph_name} not found. Skipping this graph.")
                continue

            features_cat = torch.tensor([product_features_cat], dtype=torch.int)
            features_num = torch.tensor([product_features_num], dtype=torch.float)

            if os.path.exists(label_path):
                y = self._load_label(label_path, desired_keys_labels)
                y_hat = self._load_label(label_path, desired_keys_calculated)
            else:
                print(f"Label file for {graph_name} not found. Skipping this graph.")
                continue
            # Convert to PyTorch Geometric Data
            edge_index = torch.tensor(graph_edges, dtype=torch.long)
            x = torch.tensor(face_attr, dtype=torch.float)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            y = torch.tensor([y], dtype=torch.float)
            y_hat = torch.tensor([y_hat], dtype=torch.float)
            face_grid = torch.tensor(face_grid, dtype=torch.float)
            edge_grid = torch.tensor(edge_grid, dtype=torch.float)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                y_hat=y_hat,
                features_categorical=features_cat,
                features_numerical=features_num,
                graph_name=graph_name,
                face_grid=face_grid,
                edge_grid=edge_grid,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                continue

            torch.save(data, osp.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_paths[idx], weights_only=False)
        return data
