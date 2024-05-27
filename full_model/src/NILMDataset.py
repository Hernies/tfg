import os
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.utils

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # Load class names from classes
        with open(os.path.join(root_dir, 'meta', 'dataset_classes'), 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Load house classes mapping from house_classes
        self.house_classes = {}
        with open(os.path.join(root_dir, 'meta', 'house_classes'), 'r') as f:
            for line in f:
                parts = line.strip().split()
                item_name = parts[0]
                classes = parts[1].split(',')
                self.house_classes[item_name] = classes

        # Find all house files in the meta directory
        self.house_files = sorted([
            f for f in os.listdir(os.path.join(root_dir, 'meta'))
            if f.startswith('house') and f.endswith('_files')
        ])

        # Calculate the total number of items in the dataset
        self.num_samples = 0
        for house_file in self.house_files:
            with open(os.path.join(root_dir, 'meta', house_file), 'r') as f:
                self.num_samples += len(f.readlines())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Determine which house file and line within that file corresponds to the index
        cumulative_length = 0
        for house_file in self.house_files:
             with open(os.path.join(self.root_dir, 'meta', house_file), 'r') as f:
                lines = f.readlines()
                if cumulative_length + len(lines) > idx:
                    line_idx = idx - cumulative_length
                    prefix_to_match = lines[line_idx].strip().split('_')[0]+'_'
                    break
                cumulative_length += len(lines)

        house_id = house_file.split('_')[0]
        
        images = []
        onehot_path = None

        # Generate a list of all the files in the house directory that match the prefix_to_match
        matching_files = [f for f in os.listdir(os.path.join(self.root_dir, house_id)) if f.startswith(prefix_to_match)]

        # Look for duplicates and remove them from the list
        matching_files = list(set(matching_files))

        # If there are over 11 files, print the contents of the list
        if len(matching_files) > 11:
            print(matching_files)

        # Iterate over the files in the list
        tensor_images = None
        for file in matching_files:
            if not file.endswith(('.png', '.jpg')):
                onehot_path = os.path.join(self.root_dir, house_id, file)
                continue
            if file.endswith(('_gaf.png', '.jpg')):
                img_path = os.path.join(self.root_dir, house_id, file)
                try:
                    image = Image.open(img_path)
                except Exception as e:
                    print(f"Error opening image: {e}")
                    continue
                if self.transform:
                    image = self.transform(image)
                else:
                    # Apply default transformation to ensure size consistency
                    transform = transforms.Compose([
                        transforms.Resize((256, 256)),  # Ensure all images are resized to 256x256
                        transforms.ToTensor(),
                    ])
                    image = transform(image)
                images.append(image)
        # turn images into tensor
        if not images:
            print(f"No images found for index {idx}")
        else:
            tensor_images = torch.stack(images)



        # Retrieve class information
        # house prefix is the first character and the number of the house
        house_prefix = house_id[0]+re.sub("\D", "", house_id)
        house_classes = self.house_classes[house_prefix]
        #convert house classes into int list
        try:
            house_classes = [int(i) for i in house_classes]
        except ValueError:
            print("Error converting house classes to integers")
            #print house classes
            print(house_classes)
            print (house_prefix)

            house_classes = []
        #house classes into tensor
        house_classes = torch.tensor(house_classes, dtype=torch.int16)

        # Load one-hot vector
        try:
            onehot = np.loadtxt(onehot_path, dtype=int, delimiter=',')[1:]
        except ValueError:
            with open(onehot_path, 'r') as f:
                line = f.readline().strip()
                onehot = np.array([int(i) for i in line.split(',') if i])[1:]
                #array of 32 elements
                expanded_onehot = np.zeros(32)
                #position the values in the position indicated from the house_classes
                for i in range(len(house_classes)):
                    try:
                        expanded_onehot[int(house_classes[i])-1] = onehot[i]
                    except ValueError:
                        continue
                onehot = torch.tensor(expanded_onehot, dtype=torch.int16)
                # print the onehot vector only if it doesnt match the house_classes

        

        return image, onehot, house_classes

# #Define your transformations
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     ])

# # # Load the dataset
# data_dir = '/home/hernies/Documents/tfg/full_model/data/REFIT_GAF'
# dataset = CustomImageDataset(root_dir=data_dir, transform=transform)

# # Create a DataLoader with a smaller batch size
# batch_size = 500
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# # Visualize some items
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# # Get a batch of training data
# data_iter = iter(data_loader)
# images, onehots, house_classes = next(data_iter)

# # Show images
# for i in range(batch_size):
#     print(f"One-hot vector: {onehots[i].numpy()}")
#     print(f"House classes: {house_classes}")
#     imshow(torchvision.utils.make_grid(images[i]))
