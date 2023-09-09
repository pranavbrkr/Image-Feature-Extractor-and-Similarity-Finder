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
  global caltectDataset
  caltectDataset = torchvision.datasets.Caltech101(root='./Dataset/', transform=transform_tensor, download=True)

grid_width, grid_height = 30, 10
grid_num_x = 10
grid_num_y = 10

def extractCM10x10(image_number):
  (image_tensor, label) = caltectDataset[image_number]
  color_moments = []

  image_tensor = torchvision.transforms.Resize((300, 100), antialias=True) (image_tensor)

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

  color_moments = torch.flatten(torch.tensor(color_moments))

  return color_moments


def extractHOG(image_number):

  hog_descriptor = []
  (image, label) = caltectDataset[image_number]
  image_tensor = torchvision.transforms.Resize((300, 100), antialias=True) (image)
  image_grayscale_tensor = torchvision.transforms.functional.rgb_to_grayscale(img=image_tensor).squeeze(0)

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

  hog_descriptor = torch.flatten(torch.tensor(np.array(hog_descriptor)))

  return hog_descriptor


def hook_fn(module, input, output):
    global layer_output
    layer_output = output

def extractResnetAvgpool1024(image_number):

  model = resnet50(weights=ResNet50_Weights.DEFAULT)
  (image_tensor, label) = caltectDataset[image_number]

  image_tensor = torchvision.transforms.Resize((224, 224), antialias=True) (image_tensor)


  avgpool_layer = model.avgpool
  hook = avgpool_layer.register_forward_hook(hook_fn)

  output = model(image_tensor.unsqueeze(0))

  avgpool_output_matrix = torch.squeeze(layer_output)
  avgpool_feature_descriptor = torch.tensor([((avgpool_output_matrix[i].item() + avgpool_output_matrix[i+1].item()) / 2) for i in range(0, len(avgpool_output_matrix), 2)])

  return avgpool_feature_descriptor

        
def extractResnetLayer3(image_number):

  model = resnet50(weights=ResNet50_Weights.DEFAULT)

  (image_tensor, label) = caltectDataset[image_number]
  image_tensor = torchvision.transforms.Resize((224, 224), antialias=True) (image_tensor)

  layer3_layer = model.layer3
  hook = layer3_layer.register_forward_hook(hook_fn)

  output = model(image_tensor.unsqueeze(0))

  layer3_feature_descriptor_matrix = torch.squeeze(layer_output)


  layer3_feature_descriptor = []
  
  for layer3_len in range(len(layer3_feature_descriptor_matrix)):
    layer3_feature_descriptor.append(np.average(layer3_feature_descriptor_matrix[layer3_len].detach().numpy()))

  layer3_feature_descriptor = torch.tensor(layer3_feature_descriptor)

  return layer3_feature_descriptor
    
        
def extractResnetFc(image_number):

  model = resnet50(weights=ResNet50_Weights.DEFAULT)

  (image_tensor, label) = caltectDataset[image_number]
  image_tensor = torchvision.transforms.Resize((224, 224), antialias=True) (image_tensor)

  fc_layer = model.fc
  hook = fc_layer.register_forward_hook(hook_fn)

  output = model(image_tensor.unsqueeze(0))

  fc_feature_descriptor = torch.squeeze(layer_output)

  return fc_feature_descriptor


def main():

  loadDataset()

  image_id = int(input("Enter image ID: "))

  print("1: Color moments")
  print("2: HOG")
  print("3: Resnet50 Avgpool layer")
  print("4: Resnet50 Layer 3")
  print("5: Resnet50 FC layer")
  k = int(input("Enter which feature model to extract: "))

  match k:

    case 1:
      feature_descriptor = extractCM10x10(image_id)
      print("Color moments feature descriptor is as follows:")
      print(feature_descriptor)

    case 2:
      feature_descriptor = extractHOG(image_id)
      print("HOG feature descriptor is as follows:")
      print(feature_descriptor)

    case 3:
      feature_descriptor = extractResnetAvgpool1024(image_id)
      print("Resnet50 Avgpool feature descriptor is as follows:")
      print(feature_descriptor)

    case 4:
      feature_descriptor = extractResnetLayer3(image_id)
      print("Resnet50 Layer3 feature descriptor is as follows:")
      print(feature_descriptor)

    case 5:
      feature_descriptor = extractResnetFc(image_id)
      print("Resnet50 FC feature descriptor is as follows:")
      print(feature_descriptor)

    case _:
      print("Wrong selection. Please select one of the above options.")

if __name__ == "__main__":
   main()