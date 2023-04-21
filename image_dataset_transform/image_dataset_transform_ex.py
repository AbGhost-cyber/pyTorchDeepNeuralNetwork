import torch
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib.pyplot import imshow
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as transforms

torch.manual_seed(0)


def show_data(data_sample, shape=(28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])


directory = ""
csv_file = 'index.csv'
csv_path = os.path.join(directory, csv_file)
data_name = pd.read_csv(csv_path)


# image_name = data_name.iloc[1, 1]
# image_path = os.path.join(img_directory, image_name)
# image = Image.open(image_path)
# plt.imshow(image, cmap='gray', vmin=0, vmax=255)
# plt.title(data_name.iloc[1, 0])
# plt.show()


class Dataset(Dataset):

    def __init__(self, csv_file, data_dir, transform=None):
        # Image directory
        self.data_dir = data_dir
        # transform to be used
        self.transform = transform
        data_dircsv_file = os.path.join(self.data_dir, csv_file)
        # Load the CSV file contains image info
        self.data_name = pd.read_csv(data_dircsv_file)

        # no of images in dataset
        self.len = self.data_name.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # get image file path
        img_name = os.path.join(self.data_dir, self.data_name.iloc[idx, 1])
        # open image file
        image = Image.open(img_name)

        # the class label for the image
        y = self.data_name.iloc[idx, 0]
        # apply transform if any
        if self.transform:
            image = self.transform(image)
        return image, y


dataset = Dataset(csv_file=csv_file, data_dir=directory)
sample_image = dataset[0][0]
y = dataset[0][1]

# plt.imshow(sample_image, cmap='gray', vmin=0, vmax=255)
# plt.title(y)
# plt.show()

# Combine two transforms: crop and convert to tensor. Apply the `compose` to MNIST dataset
crop_tensor_data_transform = transforms.Compose([transforms.CenterCrop(size=20), transforms.ToTensor()])
dataset = Dataset(csv_file=csv_file, data_dir=directory, transform=crop_tensor_data_transform)
print("The shape of the first element tensor: ", dataset[0][0].shape)

# Construct the `compose`. Apply it on MNIST dataset. Plot the image out.

fliptensor_data_transform = transforms.Compose([transforms.RandomVerticalFlip(p=1), transforms.ToTensor()])
dataset = Dataset(csv_file=csv_file, data_dir=directory, transform=fliptensor_data_transform)
show_data(dataset[1])

fliptensor_data_transform = transforms.Compose([transforms.RandomVerticalFlip(),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()])
dataset = Dataset(csv_file=csv_file, data_dir=directory, transform=fliptensor_data_transform)
show_data(dataset[1])
if __name__ == '__main__':
    print()
