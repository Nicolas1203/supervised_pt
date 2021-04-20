"""Utils function for data loading and data processing.
"""
import os
from torchvision import transforms
from src.datasets.YoutubeDataset import YoutubeDataset
from torch.utils.data import DataLoader

def get_transforms():
    """Define image transformations for model training and inference

    Returns: None
    """
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize((224,224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize((224,224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


def get_loaders(root_dir, annotations_file, batch_size):
    """Initialize dataset loaders for custom datasets

    Args:
        root_dir (str):             Root dir containing images/ folder and annotations.csv
        annotations_file (str) :    Name of the csv containing annotations  
        batch_size (int):           Batch size for training

    Returns:
        dict{dataloader_name:dataloader}, dict{dataset_name:dataset_size}
    """
    data_transforms = get_transforms()
    datasets = {
        'train': YoutubeDataset(
        root_dir=os.path.join(root_dir, 'images/train/'),
        csv_file=os.path.join(root_dir, annotations_file),
        transform=data_transforms['train']
        ),
        'val': YoutubeDataset(
        root_dir=os.path.join(root_dir, 'images/test/'),
        csv_file=os.path.join(root_dir, annotations_file),
        transform=data_transforms['test']
        )
    }
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=1),
        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=1)
    }
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes