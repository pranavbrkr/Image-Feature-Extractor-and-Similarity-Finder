from pymongo import MongoClient
import math
import matplotlib.pyplot as plt
from task0a import *
import scipy


client = MongoClient()
client = MongoClient(host="localhost", port=27017)

# Select the database
db = client.Multimedia_Web_DBs

caltechDataset = loadDataset()

# Fetch all documents from the collection and then sort them by "_id"
feature_descriptors = list(db.Caltech101_Feature_Descriptors.find({}))
feature_descriptors = sorted(list(db.Caltech101_Feature_Descriptors.find({})), key=lambda x: x["_id"], reverse=False)

num_labels = 101

def calculate_label_means(l, feature_model):
  
  label_vectors = [x[feature_model] for x in feature_descriptors if x["label"] == l and x["_id"] % 2 == 0]
 
  label_mean_vector = [sum(col)/len(col) for col in zip(*label_vectors)]
  return label_mean_vector


def findKRelevantImages(mean_vector, feature_model, l):

  label_vectors = [(x["_id"], x[feature_model]) for x in feature_descriptors if x["_id"] % 2 == 0]

  n = len(label_vectors)

  similarities = []

  match feature_model:

    case "color_moments":

      for i in range(n):
        similarities.append({"_id": label_vectors[i][0], "similarity": math.dist(mean_vector, label_vectors[i][1])})
      similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=False)

    case "hog":

      for i in range(n):
        similarities.append({"_id": label_vectors[i][0], "similarity": (np.dot(mean_vector, label_vectors[i][1]) / (np.linalg.norm(mean_vector) * np.linalg.norm(label_vectors[i][1])))})
      similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
    
    case "layer3" | "avgpool" | "fc":

      for i in range(n):
        similarities.append({"_id": label_vectors[i][0], "similarity": scipy.stats.pearsonr(mean_vector, label_vectors[i][1]).statistic})
      similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
    
  return similarities
      

def main():

  # Load dataset

  # User input for Image ID
  l = int(input("Enter query label: "))
  k = int(input("Enter k: "))

  features = ['color_moments', 'hog', 'layer3', 'avgpool', 'fc']

  # User input for feature model to extract
  print("1: Color moments")
  print("2: HOG")
  print("3: Resnet50 Avgpool layer")
  print("4: Resnet50 Layer 3")
  print("5: Resnet50 FC layer")
  feature_model = features[int(input("Select the feature model: ")) - 1]

  mean_vector = calculate_label_means(l, feature_model)

  similar_images = findKRelevantImages(mean_vector, feature_model, l)

  for i in range(k):
    print(similar_images[i])

  fig, axes = plt.subplots(1, k, figsize=(15, 5))

  for i in range(k):
    axes[i].imshow(caltechDataset[similar_images[i]["_id"]][1].permute(1, 2, 0))
    axes[i].set_title(f'id: {similar_images[i]["_id"]}')

  # Show the figure with all the images
  plt.show()


if __name__ == "__main__":
   main()