import json
import os
import os.path as osp
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)
from sklearn.model_selection import ParameterGrid
from torch.nn import HuberLoss, L1Loss, MSELoss, SmoothL1Loss
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from dataset import StepDataset
from models.model import GENConvRegressionModel
from src.pre_filtering import filter_dataset, normalize_numerical_features


def train(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Parameters:
    - model: The model to train.
    - train_loader: DataLoader for the training dataset.
    - optimizer: Optimizer for updating model parameters.
    - criterion: Loss function.
    - device: Device to run the training on (e.g., 'cuda' or 'cpu').

    Returns:
    - avg_loss: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        predictions = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.face_grid,
            batch.edge_grid,
            batch.features_categorical,
            batch.features_numerical,
            batch.batch,
        )
        loss = criterion(predictions, batch.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def test(model, val_loader, criterion, device):
    """
    Evaluate the model on the test dataset.

    Parameters:
    - model: The model to evaluate.
    - val_loaders: DataLoader for the test dataset.
    - criterion: Loss function.
    - device: Device to run the evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
    - avg_loss: Average test loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            predictions = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.face_grid,
                batch.edge_grid,
                batch.features_categorical,
                batch.features_numerical,
                batch.batch,
            )
            loss = criterion(predictions, batch.y.view(-1, 1))
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def get_optimizer(name, params, lr):
    name = name.lower()
    if name == "adam":
        return Adam(params, lr=lr)
    elif name == "sgd":
        return SGD(params, lr=lr)
    elif name == "rmsprop":
        return RMSprop(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def get_loss(name):
    name = name.lower()
    if name == "mse":
        return MSELoss()
    elif name == "mae":
        return L1Loss()
    elif name == "huber":
        return HuberLoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")

def calculate_metrics_estimations(dataloader):
    """
    Calculate the metrics for a model on a given dataloader.

    Parameters:
    - dataloader: DataLoader for the dataset.

    Returns:
    - losses: List of losses for each batch in the dataloader.
    """
    all_targets = []
    all_estimations = []
    for batch in dataloader:
        all_targets.append(batch.y.cpu().numpy())
        all_estimations.append(batch.y_hat.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_estimations = np.concatenate(all_estimations, axis=0)

    mae_loss = mean_absolute_error(all_targets, all_estimations)
    mse_loss = mean_squared_error(all_targets, all_estimations)
    medae_loss = median_absolute_error(all_targets, all_estimations)

    metrics = {
        "mae": mae_loss,
        "mse": mse_loss,
        "medae": medae_loss
    }

    return metrics


def calculate_metrics(model, dataloader, device):
    """
    Calculate the metrics for a model on a given dataloader.

    Parameters:
    - model: The model to evaluate.
    - dataloader: DataLoader for the dataset.
    - device: Device to run the evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
    - losses: List of losses for each batch in the dataloader.
    """
    model.eval()

    with torch.no_grad():
        all_targets = []
        all_predictions = []

        for batch in dataloader:
            batch = batch.to(device)
            predictions = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.face_grid,
            batch.edge_grid,
            batch.features_categorical,
            batch.features_numerical,
            batch.batch,
            )
            all_targets.append(batch.y.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        mae_loss = mean_absolute_error(all_targets, all_predictions)
        mse_loss = mean_squared_error(all_targets, all_predictions)
        medae_loss = median_absolute_error(all_targets, all_predictions)

        metrics = {
            "mae": mae_loss,
            "mse": mse_loss,
            "medae": medae_loss
        }

    return metrics

def train_with_early_stopping(config, train_loader, val_loaders, device):
    """
    Train the model with early stopping and hyperparameter tuning.

    Returns:
    - best_model: The model with the best performance on the test set.
    - best_loss: The lowest test loss achieved.
    - train_losses: List of training losses per epoch.
    - test_losses: List of test losses per epoch.
    """
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
        categorical_features_dim=config["model"]["categorical_features_dim"],
        numerical_features_dim=config["model"]["numerical_features_dim"],
    ).to(device)

    optimizer = get_optimizer(config["training"]["optimizer"], model.parameters(), config["training"]["lr"])
    criterion = get_loss(config["training"]["loss_fn"])

    model.apply(reset_weights)

    best_loss = float("inf")
    best_model = None
    patience_counter = 0

    train_losses = []
    test_losses = []

    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss = test(model, val_loaders, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

        if test_loss < best_loss:
            best_loss = test_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        if epoch > 0 and epoch % config['logging']['log_interval'] == 0:
            print(f"Saving model at epoch {epoch} with loss {test_loss:.4f}")
            save_path = os.path.join(config['logging']['save_dir_checkpoints'], f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(best_model)

    return model, best_loss, train_losses, test_losses


def reset_weights(m):
    """
    Reset the weights of the model to ensure a fresh start for each training run.
    """
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def get_dataloaders(config, dir_path):
    train_path = os.path.join(dir_path, config["dataset"]["train_path"])
    test_path = os.path.join(dir_path, config["dataset"]["test_path"])

    if config["dataset"]["reinitialize"]:
        print("Reinitializing datasets")

        dataset = StepDataset(
            root=train_path, reinitialize=True, transform=None, pre_transform=None, pre_filter=None, labels=config["labels"]["labels"], calculated=config["labels"]["calculated"], numerical_features=config["features"]["numerical_features"], categorical_features=config["features"]["categorical_features"]
        )
        dataset_test = StepDataset(
            root=test_path, reinitialize=True, transform=None, pre_transform=None, pre_filter=None, labels=config["labels"]["labels"], calculated=config["labels"]["calculated"], numerical_features=config["features"]["numerical_features"], categorical_features=config["features"]["categorical_features"]
        )
        print("Reinitialization complete.")

    dataset = StepDataset(
        root=train_path,
        reinitialize=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    )

    dataset_test = StepDataset(
        root=test_path,
        reinitialize=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    )

    # Concatenate both datasets
    dataset_concat = dataset
    print(f"Dataset loaded from {train_path} and {test_path}")
    return dataset_concat

if __name__ == '__main__':
    config = load_config("config.yaml")
    dir_path = os.getcwd()

    seed = config["training"]["seed"]

    figures_path = os.path.join(dir_path, config["logging"]["save_dir_figs"])
    figures_start = config['logging']['figure_start']

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

    if not os.path.exists(config['logging']['save_dir_best_model']):
        os.makedirs(config['logging']['save_dir_best_model'])

    if not os.path.exists(config['logging']['save_dir_losses']):
        os.makedirs(config['logging']['save_dir_losses'])

    if not os.path.exists(config['logging']['save_dir_checkpoints']):
        os.makedirs(config['logging']['save_dir_checkpoints'])\

    if not os.path.exists(config['logging']['save_dir_config']):
        os.makedirs(config['logging']['save_dir_config'])

    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    # Set the root directory for the datasets

    dataset_concat = get_dataloaders(config, dir_path)

    filtered_dataset, stats = filter_dataset(
        dataset_concat,
        min_num_nodes=config["filter"]["min_num_nodes"],
        max_num_nodes=config["filter"]["max_num_nodes"],
        min_label=config["filter"]["min_label"],
        max_label=config["filter"]["max_label"],
        filter_indices=config["filter"]["materials_to_ignore"],
    )
    print()
    print("====================")
    print(f"Number of graphs: {len(filtered_dataset)}")
    print(f"Statistics: {stats}")
    print(filtered_dataset[0])


    # Normalize numerical features
    if config["features"]["numerical_features"] != []:
        print("Normalizing numerical features in the datasets...")
        filtered_dataset, scaler = normalize_numerical_features(filtered_dataset)
    else:
        scaler = None
        print("No numerical features to normalize in the test dataset.")

    # Use a fixed seed via torch.manual_seed directly (not via a generator)
    # Adjust split sizes to ensure their sum matches the dataset length
    # Calculate split sizes as integers so their sum matches the dataset length
    total_len = len(filtered_dataset)
    train_size = int(config["training"]["split_ratio"] * total_len) + 1
    remaining = total_len - train_size
    val_size = remaining // 2
    test_size = remaining - val_size  # ensures sum matches total_len

    split_sizes = [train_size, val_size, test_size]
    print(f"Split sizes (train/val/test): {split_sizes}")

    train_dataset, val_dataset, test_dataset = random_split(
        filtered_dataset,
        split_sizes,
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    print(f"Number of training graphs: {len(train_dataset)}")
    print(f"Number of validation graphs: {len(val_dataset)}")
    print(f"Number of test graphs: {len(test_dataset)}")

    model, best_loss, train_loss, val_losses = train_with_early_stopping(config, train_loader, val_loader, device)
    print(f"Best validation loss: {best_loss:.4f}")

    # Save the best model
    best_model_path = os.path.join(config["logging"]["save_dir_best_model"], "best_model.pth")
    torch.save(model.state_dict(), best_model_path)
    print(f"Best model saved to {best_model_path}")

    # Save training and validation losses
    losses_df = pd.DataFrame(
        {
            "epoch": range(1, len(train_loss) + 1),
            "train_loss": train_loss,
            "val_loss": val_losses,
        }
    )
    losses_path = os.path.join(config["logging"]["save_dir_losses"], "losses.csv")
    losses_df.to_csv(losses_path, index=False)
    print(f"Training and validation losses saved to {losses_path}")


    # Get metrics for train, validation, and test datasets
    train_metrics = calculate_metrics(model, train_loader, device)
    val_metrics = calculate_metrics(model, val_loader, device)
    test_metrics = calculate_metrics(model, test_loader, device)

    train_metrics_estimations = calculate_metrics_estimations(train_loader)
    val_metrics_estimations = calculate_metrics_estimations(val_loader)
    test_metrics_estimations = calculate_metrics_estimations(test_loader)

    print("Train Metrics:", train_metrics)
    print("Validation Metrics:", val_metrics)
    print("Test Metrics:", test_metrics)

    # Save metrics to a CSV file
    metrics_df = pd.DataFrame({
        "dataset": ["train", "validation", "test"],
        "mae": [train_metrics["mae"], val_metrics["mae"], test_metrics["mae"]],
        "mse": [train_metrics["mse"], val_metrics["mse"], test_metrics["mse"]],
        "medae": [train_metrics["medae"], val_metrics["medae"], test_metrics["medae"]],
    })

    metrics_estimations_df = pd.DataFrame({
        "dataset": ["train", "validation", "test"],
        "mae_estimations": [train_metrics_estimations["mae"], val_metrics_estimations["mae"], test_metrics_estimations["mae"]],
        "mse_estimations": [train_metrics_estimations["mse"], val_metrics_estimations["mse"], test_metrics_estimations["mse"]],
        "medae_estimations": [train_metrics_estimations["medae"], val_metrics_estimations["medae"], test_metrics_estimations["medae"]],
    })

    metrics_path = os.path.join(config["logging"]["save_dir_losses"], "metrics.csv")
    metrics_estimations_path = os.path.join(config["logging"]["save_dir_losses"], "metrics_estimations.csv")
    metrics_df.to_csv(metrics_path, index=False)
    metrics_estimations_df.to_csv(metrics_estimations_path, index=False)
    print(f"Metrics saved to {metrics_path}")



    # Save the scaler
    scaler_path = os.path.join(config["logging"]["save_dir_config"], "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        torch.save(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    # Save the configuration
    config_path = os.path.join(config["logging"]["save_dir_config"], "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print("Training complete. Check the logs and saved models for details.")
    print(f"Figures will be saved in {figures_path}")

    import matplotlib.pyplot as plt

    # Get predictions and ground truths for the test set
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for batch in (test_loader):
            batch = batch.to(device)
            preds = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.face_grid,
                batch.edge_grid,
                batch.features_categorical,
                batch.features_numerical,
                batch.batch,
            )
            all_targets.append(batch.y.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(all_targets, all_predictions, alpha=0.5)
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--', label='Ideal')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title('Predictions vs Ground Truths (Test Set)')
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(figures_path, "pred_vs_gt_test.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Prediction vs Ground Truth figure saved to {fig_path}")

    # Save predictions, ground truths, and estimation errors to a CSV file
    results_df = pd.DataFrame({
        "ground_truth": all_targets,
        "prediction": all_predictions,
        "error": all_predictions - all_targets
    })
    results_csv_path = os.path.join(figures_path, "predictions_vs_ground_truths.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Predictions, ground truths, and errors saved to {results_csv_path}")



