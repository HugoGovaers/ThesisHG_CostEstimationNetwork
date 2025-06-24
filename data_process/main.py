import gc
import json
import os
import os.path as osp
import sys
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods_Solid

try:
    from data_process.graph_extractor import GraphExtractor
except ImportError:
    from .graph_extractor import GraphExtractor


def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)


def save_json_data(path_name, data):
    """Export a data to a json file"""
    with open(path_name, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)

def load_csv(file_path: str):
    """
    Loads data from a specified tab in an Excel file.

    Parameters:
    - file_path (str): Path to the Excel file.
    - sheet_name (str): Name of the sheet to load.

    Returns:
    - np.ndarray: A numpy array containing the data from the specified sheet.
    """
    try:
        # Load the specified sheet into a DataFrame
        df = pd.read_csv(file_path)
        # Convert DataFrame to numpy array
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def compute_volumes(step_path):
    # Load STEP file
    step_path = str(step_path)
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)

    if status != IFSelect_RetDone:
        raise RuntimeError("Failed to read STEP file")

    reader.TransferRoots()
    shape = reader.OneShape()

    # Total volume
    total_volume = 0.0
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solid = topods_Solid(explorer.Current())
        props = GProp_GProps()
        brepgprop_VolumeProperties(solid, props)
        total_volume += props.Mass()
        explorer.Next()

    # Bounding box
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    bounding_box_volume = dx * dy * dz

    return total_volume, bounding_box_volume


def get_value(
    df: pd.DataFrame,
    product_name: str,
    column_name: str,
    key_column: str = "Drawing ID",
):
    """
    Returns the value in `column_name` for the row where `key_column` == `product_name`.

    Args:
        df (pd.DataFrame): The dataframe to search.
        product_name (str): The identifier of the product.
        column_name (str): The name of the column whose value you want.
        key_column (str): The name of the column that holds product identifiers (default is 'graph_name').

    Returns:
        The value in the specified column, or None if not found.
    """
    row = df[df[key_column] == product_name]
    if row.empty:
        print(f"Warning: {product_name} not found in {key_column}.")
        return None  # or raise an error if you prefer
    return row.iloc[0][column_name]

def create_graph_from_step(file_path, config, raw_path):
    try:
        # Create the graph and store as .json
        extractor = GraphExtractor(file_path, config, scale_body=True)
        out = extractor.process()
        graph_index = str(file_path.stem)
        graph = [graph_index, out]

        save_json_data(osp.join(raw_path, graph_index + ".json"), graph)
        return [str(file_path.stem)]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


def create_product_feature(file_path, processed_data, config, features_path):
    product_id = str(file_path.stem)
    out = {}
    features_numerical = config["features_numerical"]
    features_categorical = config["features_categorical"]

    for feature in features_numerical:
        get_feature = get_value(processed_data, product_id, feature)
        if pd.notna(get_feature):
            out[feature] = get_feature
        else:
            print(f"Warning: {product_id} does not have {feature}.")
            out[feature] = 0
    for feature in features_categorical:
        one_hot_encoded_length = int(processed_data[feature].max()) + 1 # Allows material groups to be added in the future
        get_feature = get_value(processed_data, product_id, feature)
        one_hot_encoded = {}
        if pd.notna(get_feature):
            # Perform one-hot encoding for the feature
            for i in range(one_hot_encoded_length):  # Ensure consistent order (0, 1, 2, ...)
                key = f"{feature}_{i}"
                one_hot_encoded[key] = (int(get_feature) == i)
        else:
            print(f"Warning: {product_id} does not have {feature}.")
            # Add False for all possible one-hot encoded keys
            for i in range(one_hot_encoded_length):  # Ensure consistent order (0, 1, 2, ...)
                one_hot_encoded[f"{feature}_{i}"] = False
        out.update(one_hot_encoded)

    # Compute volumes
    try:
        total_volume, bounding_box_volume = compute_volumes(file_path)
    except Exception as e:
        total_volume = 0.0
        bounding_box_volume = 0.0

    out["total_volume"] = total_volume
    out["bounding_box_volume"] = bounding_box_volume

    wrapped_out = {product_id: out}

    save_json_data(osp.join(features_path, product_id + ".json"), wrapped_out)

def create_label(file_path, processed_data, config, labels_path):
    product_id = str(file_path.stem)
    out = {}
    labels = config["labels"]
    for label in labels:
        get_label = get_value(processed_data, product_id, label)
        if pd.notna(get_label):
            out[label] = get_label
        else:
            print(f"Warning: {product_id} does not have {label}.")
            out[label] = 0

    wrapped_out = {product_id: out}

    save_json_data(osp.join(labels_path, product_id + ".json"), wrapped_out)

def process_one_file(args):
    file_path, config, raw_path, features_path, processed_data, labels_path = args

    create_product_feature(file_path, processed_data, config, features_path)
    create_label(file_path, processed_data, config, labels_path)
    results = create_graph_from_step(file_path, config, raw_path)

    results = []

    return results


def one_hot_encode(data, column_name):
    """
    One-hot encodes a specified column in the DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame to process.
        column_name (str): The name of the column to one-hot encode.

    Returns:
        pd.DataFrame: The DataFrame with the one-hot encoded column.
    """
    one_hot = pd.get_dummies(data[column_name], prefix=column_name)
    data = data.drop(column_name, axis=1)
    return pd.concat([data, one_hot], axis=1)


def initializer(log_path=None):
    import signal
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Redirect stdout and stderr in workers
    if log_path is not None:
        sys.stdout = open(log_path, "a")
        sys.stderr = sys.stdout


if __name__ == '__main__':
    # Redirect all print and error outputs to a log file
    log_dir = "logs"

    # Generate a timestamped log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_dir, f"log_data_process{timestamp}.txt")

    # Redirect stdout and stderr to the log file
    # sys.stdout = open(log_file_path, 'w')
    # sys.stderr = sys.stdout

    attribute_config_path_os = os.path.abspath("..\\MSG_Project\\data_process\\attribute_config.json")
    attribute_config_path = Path(attribute_config_path_os)
    attribute_config = load_json(attribute_config_path)

    root_paths = []

    print(f"Logging started at {datetime.now()}")
    if attribute_config["rerun"]["train"]:
        root_paths.append(os.path.abspath("..\\MSG_Project\\data"))

    if attribute_config["rerun"]["test"]:
        root_paths.append(os.path.abspath("..\\MSG_Project\\data_test"))

    for root in root_paths:
        step_path_os = os.path.join(root, "step_files")
        raw_path_os = os.path.join(root, "raw")
        features_path_os = os.path.join(root, "features")
        labels_path_os = os.path.join(root, "labels")
        processed_data_path_os = os.path.join(root, "Processed_Data.csv")

        num_workers = 16

        processed_data = load_csv(processed_data_path_os)

        raw_path = Path(raw_path_os)
        step_path = Path(step_path_os)
        features_path = Path(features_path_os)
        labels_path = Path(labels_path_os)

        if not raw_path.exists():
            raw_path.mkdir()
            print("Creating output directory")

        step_files = list(step_path.glob("*"))

        pool = Pool(
            processes=num_workers, initializer=initializer, initargs=(log_file_path,)
        )

        try:
            results = list(tqdm.tqdm(
                pool.imap(process_one_file, zip(step_files, repeat(attribute_config), repeat(raw_path), repeat(features_path), repeat(processed_data), repeat(labels_path))),
                total=len(step_files),
                file=sys.__stdout__))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()

        pool.terminate()
        pool.join()

        graph_count = 0
        fail_count = 0
        graphs = []
        for res in results:
            if len(res) > 0:
                graph_count += 1
                graphs.append(res)
            else:
                fail_count += 1

        gc.collect()
        print(f"Process {len(results)} files. Generate {graph_count} graphs. Has {fail_count} failed files.")

class StepFileProcessor:
    def __init__(self, file_path):
        attribute_config_path_os = os.path.abspath("..\\MSG_Project\\data_process\\attribute_config.json")
        attribute_config_path = Path(attribute_config_path_os)
        self.config = load_json(attribute_config_path)
        self.file_path = Path(file_path)

    def extract_graph(self):
        extractor = GraphExtractor(self.file_path, self.config, scale_body=True)
        return extractor.process()

    def compute_volumes(self):
        total_volume, bounding_box_volume = compute_volumes(self.file_path)
        return total_volume, bounding_box_volume

    def one_hot_encode(self, data, column_name):
        """
        One-hot encodes a specified column in the DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to process.
            column_name (str): The name of the column to one-hot encode.

        Returns:
            pd.DataFrame: The DataFrame with the one-hot encoded column.
        """
        one_hot = pd.get_dummies(data[column_name], prefix=column_name)
        data = data.drop(column_name, axis=1)
        return pd.concat([data, one_hot], axis=1)

    def process(self):
        graph = self.extract_graph()
        total_volume, bounding_box_volume = self.compute_volumes()

        graph_edges = graph["graph"]["edges"]
        face_attr = graph["graph_face_attr"]
        face_grid = graph["graph_face_grid"]
        edge_attr = graph["graph_edge_attr"]
        edge_grid = graph["graph_edge_grid"]

        result = {
            "graph_edges": graph_edges,
            "face_attr": face_attr,
            "face_grid": face_grid,
            "edge_attr": edge_attr,
            "edge_grid": edge_grid,
            "total_volume": total_volume,
            "bounding_box_volume": bounding_box_volume,
        }

        return result
