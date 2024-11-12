import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Loads CSV
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # returning the total number of samples
        return len(self.annotations)

    def __getitem__(self, idx):
        # fetches the image path and label
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.annotations.iloc[idx, 1])

        # applies the transformations on the image
        if self.transform:
            image = self.transform(image)

        return image, label

# the data loader
def main():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CustomDataset(csv_file='dataset.csv', root_dir='path/to/images', transform=transform)

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    for images, labels in data_loader:
        print(f'Batch of images shape: {images.shape}')
        print(f'Batch of labels: {labels}')

if __name__ == "__main__":
    main()