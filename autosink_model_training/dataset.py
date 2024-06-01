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


