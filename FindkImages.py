from ExtractFeatureDescriptors import loadDataset
from pymongo import MongoClient
import matplotlib.pyplot as plt
import scipy
import math
import numpy as np

# Connect to MongoDB Client
client = MongoClient()
client = MongoClient(host="localhost", port=27017)

# Load image dataset
caltectDataset = loadDataset()

# Take input for image ID
image_id = int(input("Enter Image ID of the image: "))

# Take input for number "k"
k = int(input("Enter value of k: "))

# Select the database
db = client.Multimedia_Web_DBs

# Fetch all documents from the collection and then sort them by "_id"
feature_descriptors = list(db.Caltech101_Feature_Descriptors.find({}))
feature_descriptors = sorted(list(db.Caltech101_Feature_Descriptors.find({})), key=lambda x: x["_id"], reverse=False)
n = len(feature_descriptors)

euclidean_cm_distances = []
cosine_hog_distances = []
avgpool_pearson_distances = []
layer3_pearson_distances = []
fc_pearson_distances = []

for i in range(n):
  if i == image_id:
    continue

  # Calculate the relevant similarity measures

  if feature_descriptors[i].get("color_moments") and feature_descriptors[image_id].get("color_moments"):
    euclidean_cm_distances.append({"_id": feature_descriptors[i]["_id"],  "similarity": math.dist(feature_descriptors[image_id]["color_moments"], feature_descriptors[i]["color_moments"])})

  if feature_descriptors[i].get("hog") and feature_descriptors[image_id].get("hog"):
    cosine_hog_distances.append({"_id": feature_descriptors[i]["_id"],  "similarity": (np.dot(feature_descriptors[image_id]["hog"], feature_descriptors[i]["hog"]) / (np.linalg.norm(feature_descriptors[image_id]["hog"]) * np.linalg.norm(feature_descriptors[i]["hog"])))})

  if feature_descriptors[i].get("avgpool") and feature_descriptors[image_id].get("avgpool"):
    avgpool_pearson_distances.append({"_id": feature_descriptors[i]["_id"], "similarity": scipy.stats.pearsonr(feature_descriptors[image_id]["avgpool"], feature_descriptors[i]["avgpool"]).statistic})

  if feature_descriptors[i].get("layer3") and feature_descriptors[image_id].get("layer3"):
    layer3_pearson_distances.append({"_id": feature_descriptors[i]["_id"], "similarity": scipy.stats.pearsonr(feature_descriptors[image_id]["layer3"], feature_descriptors[i]["layer3"]).statistic})

  if feature_descriptors[i].get("fc") and feature_descriptors[image_id].get("fc"):
    fc_pearson_distances.append({"_id": feature_descriptors[i]["_id"], "similarity": scipy.stats.pearsonr(feature_descriptors[image_id]["fc"], feature_descriptors[i]["fc"]).statistic})


# Sort the similarity lists
euclidean_cm_similarity = sorted(euclidean_cm_distances, key=lambda x: x["similarity"], reverse=False)
cosine_hog_similarity = sorted(cosine_hog_distances, key=lambda x: x["similarity"], reverse=True)
avgpool_pearson_similarity = sorted(avgpool_pearson_distances, key=lambda x: x["similarity"], reverse=True)
layer3_pearson_similarity = sorted(layer3_pearson_distances, key=lambda x: x["similarity"], reverse=True)
fc_pearson_similarity = sorted(fc_pearson_distances, key=lambda x: x["similarity"], reverse=True)

figure_data = [
  euclidean_cm_similarity,
  cosine_hog_similarity,
  avgpool_pearson_similarity,
  layer3_pearson_similarity,
  fc_pearson_similarity,
]

#
#
# Display the similar images into a matplotlib grid
num_rows = 6
fig, axes = plt.subplots(num_rows, k, figsize=(10, 10))  # Adjust the figsize as needed

plot_titles = [
  "CM - Euclidean",
  "HOG - Cosine",
  "Avgpool - Pearson",
  "Layer 3 - Pearson",
  "Fc - Pearson",
]

for i in range(6):
    for j in range(k):
      
      if i == 0:
        if j == 0:
          axes[i, j].imshow(caltectDataset[image_id][1].permute(1, 2, 0))
          axes[i,j].set_title(f"Input image: {image_id}")
        axes[i, j].axis('off')
        continue
      else:
        axes[i, j].axis('off')
        if figure_data[i - 1]:
          axes[i, j].imshow(caltectDataset[figure_data[i - 1][j]["_id"]][1].permute(1, 2, 0))
          if j==0:
            axes[i, j].set_title(f"{plot_titles[i-1]}=> id: {caltectDataset[figure_data[i - 1][j]['_id']][0]}, sim: {round(figure_data[i - 1][j]['similarity'], 5)}", fontsize=5)
          else:
            axes[i, j].set_title(f"id: {caltectDataset[figure_data[i - 1][j]['_id']][0]}, sim: {round(figure_data[i - 1][j]['similarity'], 5)}", fontsize=5)

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.4, hspace=0.8)
plt.show()