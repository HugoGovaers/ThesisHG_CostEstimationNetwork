# 🛠️ Cost Estimation Network (CNC Product-Based Regression)

This repository provides an implementation of a machine learning pipeline for predicting **machining time** and **programming time** using material properties and product geometry extracted from STEP files.

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

⚠️ **Note:** Make sure to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) if you want to train the model on a GPU.

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
- Calculated target variable(s): `Calculated Programming Time`, `Calculated Programming Time` (to compare results with other methods on historical data)

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

1. Configure model and training parameters in `config.yaml`.
2. Run the training script:

```bash
python training.py
```

---

## 🔮 Prediction

1. Place the STEP file to be predicted in the `predict/` folder.
2. Run the prediction script:

```bash
python predict.py
```

3. Follow the prompts:
   - Enter the filename (without the `.step` extension).
   - Enter any additional features manually (Note: If feature layout changes, you must update `predict.py` accordingly).

---

## 📌 Notes

- This project assumes a high-mix, low-volume production setup.
- Ensure consistent file naming between STEP files and the `Drawing ID` column.
- You can extend the model to predict different or multiple regression targets with minimal changes.
