import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def convert_to_uci_format(path, output_dir,dataset_name, target_index = -1, version="original",n_splits=10, test_size=0.1, random_seed=0,sheet_name="", data_type="csv", delimiter=' '):
    output_dir = os.path.join(output_dir, dataset_name)
    if data_type in ["csv","txt"]:
        df = pd.read_csv(os.path.join(output_dir, path), sep=delimiter,quotechar='"', skiprows=1, on_bad_lines='skip')
        data = df.values
    elif data_type == "excel":
        df = pd.read_excel(path, sheet_name=sheet_name)
        data = df.values
    else:
        raise ValueError("The data must be in csv, txt or excel format")
    # Move the target column to the end
    if target_index != - 1:
        print(f"Moving column {target_index} to the last position.")
        features = np.delete(data, target_index, axis=1)
        target = data[:, [target_index]]
        data = np.hstack([features, target])
    # Create directories
    version_dir = os.path.join(output_dir, version)
    os.makedirs(version_dir, exist_ok=True)
    # Save data.txt
    data_txt_path = os.path.join(output_dir, "data.txt")
    np.savetxt(data_txt_path, data, fmt="%.6f")
    print(f"Saved full dataset to {data_txt_path}")
    # Generate train/test splits
    n_samples = data.shape[0]
    for split_id in range(n_splits):
        indices = np.arange(n_samples)
        idx_train, idx_test = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_seed + split_id
        )
        train_path = os.path.join(version_dir, f"index_train_{split_id}.txt")
        test_path = os.path.join(version_dir, f"index_test_{split_id}.txt")
        np.savetxt(train_path, idx_train, fmt="%d")
        np.savetxt(test_path, idx_test, fmt="%d")
        print(f"Split {split_id}: Saved to {train_path}, {test_path}")