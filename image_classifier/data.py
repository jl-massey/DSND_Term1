from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils import get_max_workers


def count_classes(data_dir):
    """
    Uses ImageFolder to check the number of classes in the provided directory.
    :param data_dir: Image directory (E.g. training)
    :return: Number of classes
    """
    return len(ImageFolder(data_dir).classes)


def class_to_idx(data_dir):
    return ImageFolder(data_dir).class_to_idx


def get_transform(arch, training=True):
    """
    Create image transformer appropriate for the model
    architecture and the desired usage (train vs. eval)
    :param arch: Model architecture.
    :param training: True if the transform is for a training dataset.
    :return: torchvision image transformer
    """
    if arch.startswith('inception'):
        cropsize = 299
    else:
        cropsize = 224
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if training:
        return transforms.Compose([
            transforms.RandomRotation(30),
            transforms.Resize(cropsize + 32),
            transforms.CenterCrop(cropsize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        return transforms.Compose([
            transforms.Resize(cropsize + 32),
            transforms.CenterCrop(cropsize),
            transforms.ToTensor(),
            normalize])


def get_dataloader(img_dir, arch, batch_size, training=True):
    """
    Creates a dataloader for an image dataset.
    :param img_dir: str: path to the directory containing class directories
    :param arch: str: valid model architecture being used
    :param training: bool: True if the data loader is for training data
    :param batch_size: Number of images to return in a batch
    :return: torch.utils.data.DataLoader
    """
    ds = ImageFolder(root=img_dir, transform=get_transform(arch, training))
    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=training, num_workers=get_max_workers())
    return dl
