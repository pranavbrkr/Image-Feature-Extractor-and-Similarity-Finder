from ExtractFeatureDescriptors import loadDataset
from statistics import covariance
from pymongo import MongoClient
import matplotlib.pyplot as plt
import scipy
import math
import numpy as np

client = MongoClient()
client = MongoClient(host="localhost", port=27017)

caltectDataset = loadDataset()

image_id = int(input("Enter Image ID of the image: "))

k = int(input("Enter value of k: "))

db = client.Multimedia_Web_DBs

feature_descriptors = list(db.Feature_Descriptors.find({}))
n = len(feature_descriptors)

euclidean_distances = []
cosine_distances = []
avgpool_intersection_distances = []
layer3_intersection_distances = []
pearson_correlation = []

for i in range(n):
  if i == image_id:
    continue

  if feature_descriptors[i].get("color_moments"):
    euclidean_distances.append({"_id": feature_descriptors[i]["_id"],  "similarity": math.dist(feature_descriptors[image_id]["color_moments"], feature_descriptors[i]["color_moments"])})

  if feature_descriptors[i].get("hog"):
    cosine_distances.append({"_id": feature_descriptors[i]["_id"],  "similarity": (np.dot(feature_descriptors[image_id]["hog"], feature_descriptors[i]["hog"]) / (np.linalg.norm(feature_descriptors[image_id]["hog"]) * np.linalg.norm(feature_descriptors[i]["hog"])))})

  if feature_descriptors[i].get("avgpool"):
    imin = sum([min(a,b) for (a,b) in zip(feature_descriptors[image_id]["layer3"], feature_descriptors[i]["layer3"])])
    imax = sum([max(a,b) for (a,b) in zip(feature_descriptors[image_id]["layer3"], feature_descriptors[i]["layer3"])])
    avgpool_intersection_distances.append({"_id": feature_descriptors[i]["_id"], "similarity": imin/imax})

  if feature_descriptors[i].get("layer3"):
    imin = sum([min(a,b) for (a,b) in zip(feature_descriptors[image_id]["layer3"], feature_descriptors[i]["layer3"])])
    imax = sum([max(a,b) for (a,b) in zip(feature_descriptors[image_id]["layer3"], feature_descriptors[i]["layer3"])])
    layer3_intersection_distances.append({"_id": feature_descriptors[i]["_id"], "similarity": imin/imax})

  if feature_descriptors[i].get("fc"):
    pearson_correlation.append({"_id": feature_descriptors[i]["_id"], "similarity": scipy.stats.pearsonr(feature_descriptors[image_id]["fc"], feature_descriptors[i]["fc"]).statistic})


euclidean_similarity = sorted(euclidean_distances, key=lambda x: x["similarity"], reverse=False)
cosine_similarity = sorted(cosine_distances, key=lambda x: x["similarity"], reverse=True)
avgpool_intersection_similarity = sorted(avgpool_intersection_distances, key=lambda x: x["similarity"], reverse=True)
layer3_intersection_similarity = sorted(layer3_intersection_distances, key=lambda x: x["similarity"], reverse=True)
pearson_similarity = sorted(pearson_correlation, key=lambda x: x["similarity"], reverse=True)

figure_data = [
  euclidean_similarity,
  cosine_similarity,
  avgpool_intersection_similarity,
  layer3_intersection_similarity,
  pearson_similarity,
]


num_rows = 6
fig, axes = plt.subplots(num_rows, k, figsize=(10, 10))  # Adjust the figsize as needed

for i in range(6):
    for j in range(k):
        
        if i == 0:
          if j == 0:
            axes[i, j].imshow(caltectDataset[image_id][1].permute(1, 2, 0))
            axes[i,j].set_title("Input image")
          axes[i, j].axis('off')
          continue

        axes[i, j].imshow(caltectDataset[figure_data[i - 1][j]["_id"]][1].permute(1, 2, 0))
        axes[i, j].axis('off')
        axes[i, j].set_title(round(figure_data[i - 1][j]["similarity"], 5))

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()

# plt.imshow(caltectDataset[image_id][1].permute(1, 2, 0))
# plt.show()

# # for index in range(k):
# #   print(euclidean_similarity[index])
# #   plt.imshow(caltectDataset[euclidean_similarity[index]["_id"]][1].permute(1, 2, 0))
# #   plt.show()

# # for index in range(k):
# #   print(cosine_similarity[index])
# #   plt.imshow(caltectDataset[cosine_similarity[index]["_id"]][1].permute(1, 2, 0))
# #   plt.show()

# # for index in range(k):
# #   print(intersection_similarity[index])
# #   plt.imshow(caltectDataset[intersection_similarity[index]["_id"]][1].permute(1, 2, 0))
# #   plt.show()

# # for index in range(k):
# #   print(pearson_similarity[index])
# #   plt.imshow(caltectDataset[pearson_similarity[index]["_id"]][1].permute(1, 2, 0))
# #   plt.show()