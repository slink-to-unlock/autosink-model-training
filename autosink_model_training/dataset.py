# dataset.py

from datasets import load_dataset

def load_custom_dataset(path_dataset_dir):
    """
    Loads a dataset from the specified directory.

    Parameters:
    path_dataset_dir (str): The path to the directory containing the dataset.

    Returns:
    Dataset: The loaded dataset.
    """
    dataset = load_dataset('imagefolder', data_dir=path_dataset_dir)
    return dataset

if __name__ == '__main__':
    import os

    # Ensure that PATH_DATASET_DIR environment variable is set
    path_dataset_dir = os.getenv('data_dir')
    if path_dataset_dir is None:
        raise ValueError("Environment variable PATH_DATASET_DIR is not set")

    dataset = load_custom_dataset(path_dataset_dir)
    print(dataset)
