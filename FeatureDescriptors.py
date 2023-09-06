from scipy.signal import convolve
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageOps

transform = transforms.Compose([
    transforms.Resize((300, 100)),  # Resize the image
    transforms.ToTensor(),           # Convert PIL image to PyTorch tensor (0-1 range)
])

transform_grayscale = transforms.Compose([
    transforms.Resize((300, 100)),  # Resize the image
])

# Use args for path later
caltectDataset = torchvision.datasets.Caltech101(root='./Dataset/', transform=transform, download=True)

datasetLength = len(caltectDataset)
resized_data = []
grid_width, grid_height = 30, 10
grid_num_x = 10
grid_num_y = 10


def extractCM10x10():
  for image_number in range(1):
    feature_descriptor = []
    (image_tensor, label) = caltectDataset[image_number]
    # image_tensor = image_tensor.permute(1, 2, 0)
    color_moments = []

    
    for i in range(grid_num_x):
      for j in range(grid_num_y):
        extracted_tensor = image_tensor[:, i * grid_width : (i + 1) * grid_width, j * grid_height : (j + 1) * grid_height]
        print(extracted_tensor)
        print(extracted_tensor.shape)

        grid_moment = []

        for channel in range(3):

          print(extracted_tensor[channel])
          print(extracted_tensor[channel].shape)


          mean = torch.mean(extracted_tensor[channel])
          std = torch.std(extracted_tensor[channel])
          skewness = torch.mean((extracted_tensor[channel] - mean) ** 3) / std ** 3

          grid_moment.append([mean, std, skewness])

        color_moments.append(grid_moment)

    print(len(color_moments))
    color_moments = torch.tensor(color_moments)
    print(color_moments.shape)


def extractHOG():
  caltectDataset = torchvision.datasets.Caltech101(root='./Dataset/', transform=transform_grayscale, download=True)
  for image_number in range(1):
    hog_descriptor = []
    (image, label) = caltectDataset[image_number]
    image_grayscale = ImageOps.grayscale(image)
    # image_grayscale.show()
    image_grayscale_tensor = transforms.functional.pil_to_tensor(image_grayscale)
    image_grayscale_tensor = image_grayscale_tensor.permute(1, 2, 0)[:, :, -1]
    print(image_grayscale_tensor.shape)
    print(image_grayscale_tensor)
    # plt.imshow(image_grayscale_tensor)
    # plt.show()

    count = 0
    for i in range(grid_num_x):
      for j in range(grid_num_y):
        # print(f"{i * grid_width} : {(i + 1) * grid_width}, {j * grid_height} : {(j + 1) * grid_height}")
        # extracted_tensor = image_grayscale_tensor[i * grid_width : (i + 1) * grid_width, j * grid_height : (j + 1) * grid_height]
        extracted_tensor = image_grayscale_tensor[i * grid_width : (i + 1) * grid_width, j * grid_height : (j + 1) * grid_height]
        # print(extracted_tensor.shape)


        Gx = convolve(extracted_tensor, torch.tensor([[-1, 0, 1]]), "same")
        Gy = convolve(extracted_tensor, torch.tensor([[-1], [0], [1]]), "same")

        grad_magnitude = []
        grad_angle = []
        for a in range(len(Gx)):
          for b in range(len(Gx[0])):
            grad_magnitude.append(np.sqrt(np.square(Gx[a][b]) + np.square(Gy[a][b])))
            grad_angle.append(int((np.arctan2(Gy[a][b], Gx[a][b])) * (180 / np.pi) % 360))

        bins = np.zeros(9)
        
        for grad_index in range(len(grad_magnitude)):
          bins[(grad_angle[grad_index] // 40)] += grad_magnitude[grad_index]
        

        hog_descriptor.append(bins)

    hog_descriptor = torch.tensor(hog_descriptor)
    print(hog_descriptor)
    print(hog_descriptor.shape)
        



# extractCM10x10()
extractHOG()