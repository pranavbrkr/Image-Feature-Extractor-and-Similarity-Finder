from ExtractFeatureDescriptors import loadDataset
from pymongo import MongoClient
import matplotlib.pyplot as plt
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

for i in range(n):
  if i == image_id:
    continue

  euclidean_distances.append({"id": feature_descriptors[i]["_id"],  "similarity": math.dist(feature_descriptors[image_id]["hog"], feature_descriptors[i]["hog"])})
  cosine_distances.append({"id": feature_descriptors[i]["_id"],  "similarity": (np.dot(feature_descriptors[image_id]["hog"], feature_descriptors[i]["hog"]) / (np.linalg.norm(feature_descriptors[image_id]["hog"]) * np.linalg.norm(feature_descriptors[i]["hog"])))})

print(euclidean_distances)
print(cosine_distances)

euclidean_similarity = sorted(euclidean_distances, key=lambda x: x["similarity"], reverse=False)
cosine_similarity = sorted(cosine_distances, key=lambda x: x["similarity"], reverse=False)

print(euclidean_similarity)
print(cosine_similarity)
plt.imshow(caltectDataset[image_id][1].permute(1, 2, 0))
plt.show()

for index in range(k):
  print(euclidean_similarity[index])
  plt.imshow(caltectDataset[euclidean_similarity[index]["id"]][1].permute(1, 2, 0))
  plt.show()
  print(cosine_similarity[index])
  plt.imshow(caltectDataset[cosine_similarity[index]["id"]][1].permute(1, 2, 0))
  plt.show()