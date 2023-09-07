from scipy.signal import convolve
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from PIL import ImageOps

# Resize image to 300x100 and convert to tensor
transform_300x100 = transforms.Compose([
  transforms.Resize((300, 100)),
  transforms.ToTensor(),
])

# Transform image to grayscale and resize to 300x100
transform_grayscale = transforms.Compose([
  transforms.Resize((300, 100)),
])

# Resize image to 224x224 and convert to tensor
transform_224x224 = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
])

# Load Caltech Dataset
caltectDataset = torchvision.datasets.Caltech101(root='./Dataset/', transform=transform_300x100, download=True)

datasetLength = len(caltectDataset)
resized_data = []
grid_width, grid_height = 30, 10
grid_num_x = 10
grid_num_y = 10


def extractCM10x10():
  for image_number in range(1):
    (image_tensor, label) = caltectDataset[image_number]
    color_moments = []

    # Extract 30x10 grid from the image    
    for i in range(grid_num_x):
      for j in range(grid_num_y):
        extracted_tensor = image_tensor[:, i * grid_width : (i + 1) * grid_width, j * grid_height : (j + 1) * grid_height]

        grid_moment = []

        for channel in range(3):

          # Calculate color moments metrics
          mean = torch.mean(extracted_tensor[channel])
          std = torch.std(extracted_tensor[channel])
          skewness = torch.mean((extracted_tensor[channel] - mean) ** 3) / std ** 3

          grid_moment.append([mean, std, skewness])

        color_moments.append(grid_moment)

    color_moments = torch.tensor(color_moments)
    print("**********")
    print("Following is a color moments feature descriptor")
    print(color_moments)
    print(color_moments.shape)
    print("**********")


def extractHOG():

  caltectDataset = torchvision.datasets.Caltech101(root='./Dataset/', transform=transform_grayscale, download=True)

  for image_number in range(1):
    hog_descriptor = []
    (image, label) = caltectDataset[image_number]
    image_grayscale = ImageOps.grayscale(image)

    image_grayscale_tensor = transforms.functional.pil_to_tensor(image_grayscale)
    image_grayscale_tensor = image_grayscale_tensor.permute(1, 2, 0)[:, :, -1]

    for i in range(grid_num_x):
      for j in range(grid_num_y):

        extracted_tensor = image_grayscale_tensor[i * grid_width : (i + 1) * grid_width, j * grid_height : (j + 1) * grid_height]

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
    print("**********")
    print("Following is a HOG feature descriptor")
    print(hog_descriptor)
    print(hog_descriptor.shape)
    print("**********")


def hook_fn(module, input, output):
    global layer_output
    layer_output = output

def extractResnetAvgpool1024():
  caltectDataset = torchvision.datasets.Caltech101(root='./Dataset/', transform=transform_224x224, download=True)
  model = resnet50(weights=ResNet50_Weights.DEFAULT)

  for image_num in range(1):
    (image_tensor, label) = caltectDataset[image_num]


    avgpool_layer = model.avgpool
    hook = avgpool_layer.register_forward_hook(hook_fn)

    output = model(image_tensor.unsqueeze(0))

    avgpool_output_matrix = torch.squeeze(layer_output)
    avgpool_feature_descriptor = torch.tensor([((avgpool_output_matrix[i].item() + avgpool_output_matrix[i+1].item()) / 2) for i in range(0, len(avgpool_output_matrix), 2)])
    print("**********")
    print("Following is a avgpool 1024 feature descriptor")
    print(avgpool_feature_descriptor)
    print(avgpool_feature_descriptor.shape)
    print("**********")

        
def extractResnetLayer3():
  caltectDataset = torchvision.datasets.Caltech101(root='./Dataset/', transform=transform_224x224, download=True)
  model = resnet50(weights=ResNet50_Weights.DEFAULT)

  for image_num in range(1):
    (image_tensor, label) = caltectDataset[image_num]

    layer3_layer = model.layer3
    hook = layer3_layer.register_forward_hook(hook_fn)

    output = model(image_tensor.unsqueeze(0))

    layer3_feature_descriptor_matrix = torch.squeeze(layer_output)


    layer3_feature_descriptor = []
    
    for layer3_len in range(len(layer3_feature_descriptor_matrix)):
      layer3_feature_descriptor.append(np.average(layer3_feature_descriptor_matrix[layer3_len].detach().numpy()))

    layer3_feature_descriptor = torch.tensor(layer3_feature_descriptor)

    print("**********")
    print("Following is a layer 3 feature descriptor")
    print(layer3_feature_descriptor)
    print(layer3_feature_descriptor.shape)
    print("**********")
    
        
def extractResnetFc():
  caltectDataset = torchvision.datasets.Caltech101(root='./Dataset/', transform=transform_224x224, download=True)
  model = resnet50(weights=ResNet50_Weights.DEFAULT)

  for image_num in range(1):
    (image_tensor, label) = caltectDataset[image_num]

    fc_layer = model.fc
    hook = fc_layer.register_forward_hook(hook_fn)

    output = model(image_tensor.unsqueeze(0))

    fc_feature_descriptor = torch.squeeze(layer_output)

    print("**********")
    print("Following is a fc feature descriptor")
    print(fc_feature_descriptor)
    print(fc_feature_descriptor.shape)
    print("**********")



extractCM10x10()
extractHOG()
extractResnetAvgpool1024()
extractResnetLayer3()
extractResnetFc()