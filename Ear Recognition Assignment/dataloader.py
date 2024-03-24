from torchvision import transforms, datasets
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image
import sys, copy, os

class CustomImageDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)
        self.labels = [int(img.split('-')[0]) for img in self.all_imgs]

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def create_dataloaders(train_dir, val_dir, test_dir, batch_size=16):
    transform_train = transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomRotation(30),
                        transforms.CenterCrop(224),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                        transforms.ToTensor(),
                        # transforms.RandomHorizontalFlip(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform = transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the training dataset without transforms
    full_train_dataset = CustomImageDataset(train_dir)
    # test_dataset = CustomImageDataset(test_dir, transform=transform)

    # Instead of using the val folder, split the train into train and val (because val has different classes)
    # Split the dataset into training and validation sets
    
    uniq_ids = set(full_train_dataset.labels)
    translation_map = {value: index for index, value in enumerate(uniq_ids)}
    num_classes = len(uniq_ids) # Get the number of unique classes

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_indices, val_indices = random_split(full_train_dataset, [train_size, val_size])

    # Apply transforms to training and validation subsets
    train_subset = Subset(full_train_dataset, train_indices.indices)
    train_subset.dataset = copy.deepcopy(full_train_dataset)
    train_subset.dataset.transform = transform_train

    val_subset = Subset(full_train_dataset, val_indices.indices)
    val_subset.dataset = copy.deepcopy(full_train_dataset)
    val_subset.dataset.transform = transform

    # Create DataLoaders for training, validation, and test sets
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes

if __name__ == "__main__":

    # Check if an argument is provided
    if len(sys.argv) > 1:
        train_dir = sys.argv[1]
        print("Train path provided:")
    else:
        print("No train path provided, defaulting to:")
        train_dir = os.path.join('datasets', 'ears', 'images-cropped', 'train')
        print(train_dir)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    dataset = CustomImageDataset(main_dir=train_dir, transform=transform)

    # DataLoader
    loader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False)

    # Calculate mean and std
    mean = 0.
    std = 0.
    nb_samples = 0.
    for images, _ in loader:  # Unpack the tuple
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print(f'Mean: {mean}\nStd: {std}')




