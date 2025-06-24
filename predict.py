import os
import pickle
import shutil
import sys

import pandas as pd
import torch
import yaml
from torch_geometric.data import Data

from data_process.main import StepFileProcessor
from dataset import StepDataset  # You may need to adapt this for single file
from models.model import GENConvRegressionModel


def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def load_scaler(path):
    scaler = torch.load(path, weights_only=False)
    return scaler

def load_model(config, device, model_path):
    model = GENConvRegressionModel(
        hidden_dim=config["model"]["hidden_dim"],
        face_attr_emb=config["model"]["face_attr_emb"],
        edge_attr_emb=config["model"]["edge_attr_emb"],
        face_grid_emb=config["model"]["face_grid_emb"],
        edge_grid_emb=config["model"]["edge_grid_emb"],
        dropout_face_encoder=config["model"]["dropout_face_encoder"],
        dropout_edge_encoder=config["model"]["dropout_edge_encoder"],
        dropout_face_grid_encoder=config["model"]["dropout_face_grid_encoder"],
        dropout_edge_grid_encoder=config["model"]["dropout_edge_grid_encoder"],
        dropout_graph_nn=config["model"]["dropout_graph_nn"],
        dropout_regression=config["model"]["dropout_regression"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def main():
    # Load config
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    catageorical_features_dim = config["model"]["categorical_features_dim"]
    numerical_features_dim = config["model"]["numerical_features_dim"]

    root = os.getcwd()
    predict_dir = os.path.join(root, "predict")

    step_name = input("Enter the path to the STEP file: ").strip()

    # Determine the extension of the file in predict_dir
    step_path_stp = os.path.join(predict_dir, step_name + ".stp")
    step_path_STEP = os.path.join(predict_dir, step_name + ".STEP")

    if os.path.isfile(step_path_stp):
        # Convert .stp to .STEP if .stp exists
        shutil.copyfile(step_path_stp, step_path_STEP)
        print(f"Converted {step_path_stp} to {step_path_STEP}")
        step_path = step_path_STEP
    elif os.path.isfile(step_path_STEP):
        step_path = step_path_STEP
    else:
        print(f"File not found: {step_path_stp} or {step_path_STEP}")
        return
    print(f"Using STEP file: {step_path}")

    # Ask for material group ID and quantity
    material_group_id = input("Enter material group ID: ").strip()
    quantity = input("Enter quantity: ").strip()
    try:
        material_group_id = int(material_group_id)
        quantity = float(quantity)
    except ValueError:
        print("Invalid input for material group ID or quantity.")
        return

    # Set device to GPU if available, otherwise CPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"{num_gpus} GPU(s) available.")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        torch.cuda.set_device(0)  # Only if you know GPU 0 is the desired one
        device = torch.device("cuda:0")
        print("Using GPU.")
    else:
        print("No GPU available, using CPU.")
        device = torch.device("cpu")

    # Load scaler and model
    scaler_path = os.path.join(config["logging"]["save_dir_config"], "scaler.pkl")
    model_path = os.path.join(config["logging"]["save_dir_best_model"], "best_model.pth")
    scaler = load_scaler(scaler_path)
    model = load_model(config, device, model_path)

    # Preprocess the single step file
    step_processs = StepFileProcessor(
        file_path=step_path,
    )
    data = step_processs.process()

    face_attr = data["face_attr"]
    edge_index = data["graph_edges"]
    edge_attr = data["edge_attr"]
    face_grid = data["face_grid"]
    edge_grid = data["edge_grid"]

    features_numerical = [quantity, data["total_volume"], data["bounding_box_volume"],]
    print(f"Numerical features: {features_numerical}")
    # One-hot encode material_group_id
    features_categorical = [0] * catageorical_features_dim
    if 0 <= material_group_id < catageorical_features_dim:
        features_categorical[material_group_id] = 1
    else:
        print(f"Material group ID {material_group_id} out of range (0-{catageorical_features_dim-1}).")


    features_numerical = scaler.transform([features_numerical])
    print(f"Scaled numerical features: {features_numerical}")
    features_numerical = features_numerical[0].tolist()  # Convert to list for tensor conversion

    print(f"Categorical features: {features_categorical}")
    features_categorical =torch.tensor([features_categorical], dtype=torch.int)
    features_numerical = torch.tensor([features_numerical], dtype=torch.float)
    x = torch.tensor(face_attr, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    face_grid = torch.tensor(face_grid, dtype=torch.float)
    edge_grid = torch.tensor(edge_grid, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        face_grid=face_grid,
        edge_grid=edge_grid,
        features_categorical=features_categorical,
        features_numerical=features_numerical
    )

    data = data.to(device)

    print(data)

    # Predict
    with torch.no_grad():
        prediction = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.face_grid,
            data.edge_grid,
            data.features_categorical,
            data.features_numerical,
            torch.zeros(x.size(0), dtype=torch.long, device=device)  # batch
        )
    print(f"Predicted value: {prediction.item()}")

if __name__ == "__main__":
    main()
