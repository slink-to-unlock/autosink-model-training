from transformers import AutoFeatureExtractor
from autosink_model_training.dataset import load_custom_dataset
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomHorizontalFlip, ColorJitter, RandomApply, RandomVerticalFlip

# Function to preprocess and augment images
def preprocess_images(example, transform):
    example['pixel_values'] = transform(example['image'].convert('RGB'))
    return example


def data_preprocess(dataset, feature_extractor):
    pretrained_size = feature_extractor.size['shortest_edge']
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    transform = Compose([
        Resize((pretrained_size, pretrained_size)),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomApply([ColorJitter(brightness=0.2, contrast=0.2, hue=0.2)], p=0.5),
        ToTensor(),
        normalize,
    ])
    preprocessed_dataset = dataset.map(lambda example: preprocess_images(example, transform), batched=False)
    preprocessed_dataset = preprocessed_dataset.remove_columns('image')

    return preprocessed_dataset

def data_preprocess_eval(dataset, feature_extractor):
    pretrained_size = feature_extractor.size['shortest_edge']
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    transform = Compose([
        Resize((pretrained_size, pretrained_size)),
        ToTensor(),
        normalize,
    ])
    preprocessed_dataset = dataset.map(lambda example: preprocess_images(example, transform), batched=False)
    preprocessed_dataset = preprocessed_dataset.remove_columns('image')

    return preprocessed_dataset['train']


def split_dataset(dataset):
    """
    Returns:
    tuple: The train and evaluation datasets.
    """
    # Split the dataset into training and evaluation sets
    train_test_split = dataset['train'].train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    return train_dataset, eval_dataset


if __name__ == '__main__':
    # ResNet model load
    feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/resnet-50')

    # Ensure that PATH_DATASET_DIR environment variable is set
    path_dataset_dir = 'path_to_dataset'
    if path_dataset_dir is None:
        raise ValueError("Environment variable PATH_DATASET_DIR is not set")

    dataset = load_custom_dataset(path_dataset_dir)
    print(dataset)

    preprocessed_dataset= data_preprocess(dataset, feature_extractor)
    train_dataset, eval_dataset = split_dataset(preprocessed_dataset)