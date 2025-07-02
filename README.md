# 🛠️ Cost Estimation Network (CNC Product-Based Regression)

A **PyTorch-based pipeline** for predicting **machining time** and **programming time** using material properties and product geometry extracted from STEP files.  
Designed for **high-mix, low-volume CNC production environments**, this tool enables cost estimation based on 3D product data and material metadata.

---

## 📦 Installation

We recommend using [Miniconda](https://www.anaconda.com/download/success) for environment management.

```bash
# Create and activate environment
conda create --name MSG_PyTorch python=3.10.8
conda activate MSG_PyTorch

# Install Mamba (faster Conda alternative)
conda install -n base mamba -c conda-forge

# Install dependencies
mamba install occwl -c lambouj -c conda-forge
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install tqdm torch_geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install -U scikit-learn pandas
```

⚠️ **Note:** Ensure that a compatible [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) is installed if you intend to train the model on a GPU. Otherwise, PyTorch will default to CPU execution.

---

## 📁 Data Preparation

Create the following folder structure inside your project directory:

```
MSG_Project/
├── data/
│   ├── steps/
│   ├── step_files/
│   ├── raw/
│   ├── processed/
│   ├── labels/
│   ├── features/
│   └── Processed_Data.csv
├── data_test/
│   ├── steps/
│   ├── step_files/
│   ├── raw/
│   ├── processed/
│   ├── labels/
│   ├── features/
│   └── Processed_Data.csv
```

### 🔍 `Processed_Data.csv`

This CSV file is the main metadata index and should include:

#### Required Columns:
- `Drawing ID`: Must match the corresponding STEP file name (excluding the extension).
- Target variable(s): e.g., `Programming Time`, `Machining Time`

#### Optional Columns:
- Categorical features (as integers).
- Numerical features (as floats).
- Calculated target variable(s): `Calculated Programming Time`, `Calculated Machining Time` (to compare results with other methods on historical data).
   
### 📂 `steps/` Folder

Store all STEP files here. File names must match the `Drawing ID` entries in `Processed_Data.csv`.

---

## ⚙️ Data Processing

1. Edit `data_process/attribute_config.json`:
   - Define the label name.
   - Specify which columns are categorical or numerical features.

2. Run the following command inside the created environment:

```bash
python data_process/main.py
```

---

## 🏋️ Training

1. Configure model and training parameters in `config.yaml`. Example:

```yaml

dataset:
  train_path: "data"
  test_path: "data_test"
  reinitialize: False # If True, the dataset will be reinitialized and all previous data will be lost

labels:
  labels: ["Total Time"] # Labels to be predicted
  calculated: ["Calculated Time"] # Estimations by experts

features:
  categorical_features: ["Material Group ID"]
  numerical_features: ["Quantity", "total_volume", "bounding_box_volume"] # total_volume and bounding_box_volume are obtained from step file

filter:
  min_num_nodes: 1
  max_num_nodes: 400
  min_label: 0.1
  max_label: 10
  materials_to_ignore: [0, 12, 13, 14] # 0 = "unknown", 12 = "Insert", 13 = "Procured Part", 14 = "Articles"

model:
  hidden_dim: 32
  face_attr_dim: 14
  face_attr_emb: 32
  edge_attr_dim: 15
  edge_attr_emb: 32
  face_grid_dim: 7
  face_grid_emb: 32
  edge_grid_dim: 12
  edge_grid_emb: 32
  categorical_features_dim: 18
  numerical_features_dim: 3
  dropout_face_encoder: 0.1
  dropout_edge_encoder: 0.1
  dropout_face_grid_encoder: 0.1
  dropout_edge_grid_encoder: 0.1
  dropout_graph_nn: 0.0
  dropout_regression: 0.2

training:
  epochs: 500
  patience: 50
  early_stopping: true
  batch_size: 32
  lr: 0.001
  seed: 42
  split_ratio: 0.8
  optimizer: "adam"  # Options: "adam", "sgd", "rmsprop"
  loss_fn: "huber"  # Options: "mse", "mae", "huber"

logging:
  save_dir_checkpoints: "./logs/checkpoints"
  save_dir_best_model: "./logs/best_model"
  save_dir_losses: "./logs/losses"
  save_dir_config: "./logs/config"
  save_dir_figs : "./logs/Figures/Programming"
  figure_start : "Programming_"
  log_interval: 10
```

2. Run the training script:

```bash
python training.py
```

## 🔮 Prediction

1. Place the STEP file to be predicted in the `predict/` folder.
2. Run the prediction script:

```bash
python predict.py
```

3. Follow the prompts:
   - Enter the STEP filename (without the `.step` extension).
   - Enter additional feature values if required (based on your model configuration).  
     ⚠️ **Important:** If the feature list changes (columns added, removed, or reordered in training data), you must also update `predict.py` to reflect the new layout.
---

## 📌 Notes

- This project assumes a high-mix, low-volume production setup.
- Ensure consistent file naming between STEP files and the `Drawing ID` column.
- You can extend the model to predict different or multiple regression targets with minimal changes.

---
