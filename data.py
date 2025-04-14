import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

class ResDataset(Dataset):
    def __init__(self, root, mode='train'):
        train = True if mode == 'train' else False
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [.5, .5, .5],
                std = [.5, .5, .5]
            ),
            transforms.Resize((224, 224))
        ])
        self.dataset = torchvision.datasets.CIFAR10(
            root = root,
            train = train,
            transform = transform,
            download = True
        )
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label

def get_data_loaders(train_root, test_root, batch_size=100):
    train_dataset = ResDataset(train_root, 'train')
    test_dataset = ResDataset(test_root, 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader
