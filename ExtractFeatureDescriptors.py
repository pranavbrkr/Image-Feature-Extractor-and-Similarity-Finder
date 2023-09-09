from scipy.signal import convolve
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights

# Convert images from PIL.JpegImagePlugin.JpegImageFile to Tensor
transform_tensor = transforms.Compose([
  transforms.ToTensor(),
])

# Load Caltech Dataset
def loadDataset():
  return torchvision.datasets.Caltech101(root='./Dataset/', transform=transform_tensor, download=True)

# Set grid size and loop iteration counter
grid_width, grid_height = 30, 10
grid_num_x = 10
grid_num_y = 10

# Extract Color moments feature descriptor from the image
def extractCM10x10(caltectDataset, image_number):
  (image_tensor, label) = caltectDataset[image_number]
  color_moments = []

  # Resize the tensor to 300x100
  image_tensor = torchvision.transforms.Resize((300, 100), antialias=True) (image_tensor)

  # Extract 30x10 grid from the image    
  for i in range(grid_num_x):
    for j in range(grid_num_y):
      # Select the 30x10 grid from the complete image
      extracted_tensor = image_tensor[:, i * grid_width : (i + 1) * grid_width, j * grid_height : (j + 1) * grid_height]

      grid_moment = []

      for channel in range(3):

        # Calculate color moments for each grid
        # extracted_tensor[channel] will select entire R, G, B planes
        mean = torch.mean(extracted_tensor[channel])
        std = torch.std(extracted_tensor[channel])
        skewness = torch.mean((extracted_tensor[channel] - mean) ** 3) / std ** 3

        grid_moment.append([mean, std, skewness])

      color_moments.append(grid_moment)

  color_moments = torch.flatten(torch.tensor(color_moments))

  return color_moments

# Extract HOG feature descriptor
def extractHOG(caltectDataset, image_number):

  hog_descriptor = []
  (image, label) = caltectDataset[image_number]
  
  # Resize the image to 224x224
  image_tensor = torchvision.transforms.Resize((300, 100), antialias=True) (image)
  
  # Convert RGB to Grayscale and remove outer dimension
  image_grayscale_tensor = torchvision.transforms.functional.rgb_to_grayscale(img=image_tensor).squeeze(0)

  for i in range(grid_num_x):
    for j in range(grid_num_y):

      extracted_tensor = image_grayscale_tensor[i * grid_width : (i + 1) * grid_width, j * grid_height : (j + 1) * grid_height]

      # Convolve the [-1, 0, 1] and [-1, 0, 1]T over the 30x10 grid
      Gx = convolve(extracted_tensor, torch.tensor([[-1, 0, 1]]), "same")
      Gy = convolve(extracted_tensor, torch.tensor([[-1], [0], [1]]), "same")

      grad_magnitude = []
      grad_angle = []
      for a in range(len(Gx)):
        for b in range(len(Gx[0])):
          # Calculate magnitude and angle for each pixel based on Gx and Gy
          grad_magnitude.append(np.sqrt(np.square(Gx[a][b]) + np.square(Gy[a][b])))
          grad_angle.append(int((np.arctan2(Gy[a][b], Gx[a][b])) * (180 / np.pi) % 360))

      bins = np.zeros(9)
      
      for grad_index in range(len(grad_magnitude)):
        # Add to the bins on the basis of angle
        bins[(grad_angle[grad_index] // 40)] += grad_magnitude[grad_index]
      

      hog_descriptor.append(bins)

  hog_descriptor = torch.flatten(torch.tensor(np.array(hog_descriptor)))

  return hog_descriptor


def hook_fn(module, input, output):
    global layer_output
    layer_output = output

# Extract output of Avgpool 1024 Layer output from Resnet50
def extractResnetAvgpool1024(caltectDataset, image_number):

  model = resnet50(weights=ResNet50_Weights.DEFAULT)
  (image_tensor, label) = caltectDataset[image_number]

  # Resize the image to 224x224
  image_tensor = torchvision.transforms.Resize((224, 224), antialias=True) (image_tensor)

  # Select Avgpool 1024 layer
  avgpool_layer = model.avgpool
  # Register the forward hook for the above layer
  hook = avgpool_layer.register_forward_hook(hook_fn)

  output = model(image_tensor.unsqueeze(0))

  avgpool_output_matrix = torch.squeeze(layer_output)
  avgpool_feature_descriptor = torch.tensor([((avgpool_output_matrix[i].item() + avgpool_output_matrix[i+1].item()) / 2) for i in range(0, len(avgpool_output_matrix), 2)])

  return avgpool_feature_descriptor

        
def extractResnetLayer3(caltectDataset, image_number):

  model = resnet50(weights=ResNet50_Weights.DEFAULT)

  (image_tensor, label) = caltectDataset[image_number]
  image_tensor = torchvision.transforms.Resize((224, 224), antialias=True) (image_tensor)

  # Select Layer3
  layer3_layer = model.layer3
  # Register the forward hook for the above layer
  hook = layer3_layer.register_forward_hook(hook_fn)

  output = model(image_tensor.unsqueeze(0))

  layer3_feature_descriptor_matrix = torch.squeeze(layer_output)


  layer3_feature_descriptor = []
  
  for layer3_len in range(len(layer3_feature_descriptor_matrix)):
    layer3_feature_descriptor.append(np.average(layer3_feature_descriptor_matrix[layer3_len].detach().numpy()))

  layer3_feature_descriptor = torch.tensor(layer3_feature_descriptor)

  return layer3_feature_descriptor
    
        
def extractResnetFc(caltectDataset, image_number):

  model = resnet50(weights=ResNet50_Weights.DEFAULT)

  (image_tensor, label) = caltectDataset[image_number]
  image_tensor = torchvision.transforms.Resize((224, 224), antialias=True) (image_tensor)

  # Select FC layer
  fc_layer = model.fc
  # Register the forward hook for the above layer
  hook = fc_layer.register_forward_hook(hook_fn)

  output = model(image_tensor.unsqueeze(0))

  fc_feature_descriptor = torch.squeeze(layer_output)

  return fc_feature_descriptor


def main():

  # Load dataset
  caltectDataset = loadDataset()

  # User input for Image ID
  image_id = int(input("Enter image ID: "))

  # User input for feature model to extract
  print("1: Color moments")
  print("2: HOG")
  print("3: Resnet50 Avgpool layer")
  print("4: Resnet50 Layer 3")
  print("5: Resnet50 FC layer")
  k = int(input("Enter which feature model to extract: "))

  # Switch case for feature model input
  match k:

    case 1:
      feature_descriptor = extractCM10x10(caltectDataset, image_id)
      print("Color moments feature descriptor is as follows:")
      print(feature_descriptor)

    case 2:
      feature_descriptor = extractHOG(caltectDataset, image_id)
      print("HOG feature descriptor is as follows:")
      print(feature_descriptor)

    case 3:
      feature_descriptor = extractResnetAvgpool1024(caltectDataset, image_id)
      print("Resnet50 Avgpool feature descriptor is as follows:")
      print(feature_descriptor)

    case 4:
      feature_descriptor = extractResnetLayer3(caltectDataset, image_id)
      print("Resnet50 Layer3 feature descriptor is as follows:")
      print(feature_descriptor)

    case 5:
      feature_descriptor = extractResnetFc(caltectDataset, image_id)
      print("Resnet50 FC feature descriptor is as follows:")
      print(feature_descriptor)

    case _:
      print("Wrong selection. Please select one of the above options.")

if __name__ == "__main__":
   main()