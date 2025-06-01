import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    """
    Prepares train, val, and test dataloaders using ImageFolder.

    Args:
        data_dir (str): Path to the dataset folder containing 'train', 'val', 'test' subfolders.
        batch_size (int): Batch size for loading data.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        tuple: train_loader, val_loader, test_loader
    """

    # Data augmentation for training set
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Basic transform for val/test
    test_val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Paths to subfolders
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')
    test_path = os.path.join(data_dir, 'test')

    # ImageFolder datasets
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_path, transform=test_val_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=test_val_transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader