import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

transform = transforms.Compose([
    transforms.Resize((300, 100)),  # Resize the image
    transforms.ToTensor(),           # Convert PIL image to PyTorch tensor (0-1 range)
])

# Use args for path later
caltectDataset = torchvision.datasets.Caltech101(root='./Dataset/', transform=transform, download=True)

datasetLength = len(caltectDataset)
resized_data = []
grid_dimension = 10
grid_num_x = 30
grid_num_y = 10

for i in range(1):
  feature_descriptor = []
  (image_tensor, label) = caltectDataset[i]
  # image_tensor = image_tensor.permute(1, 2, 0)
  count = 0

  
  for i in range(grid_num_x):
    for j in range(grid_num_y):
      extracted_tensor = image_tensor[:, i * grid_dimension : (i + 1) * grid_dimension, j * grid_dimension : (j + 1) * grid_dimension]
      print(extracted_tensor)
      print(extracted_tensor.shape)

      # for channel in range(3):
      #   mean = torch.mean(extracted_tensor[:][:][channel])
      #   std = torch.std(extracted_tensor[:][:][channel])
      #   skewness = torch.mean((extracted_tensor[:][:][channel] - mean) ** 3) / std ** 3