from ExtractFeatureDescriptors import loadDataset
from pymongo import MongoClient
import matplotlib.pyplot as plt
import math

client = MongoClient()
client = MongoClient(host="localhost", port=27017)

caltectDataset = loadDataset()

image_id = int(input("Enter Image ID of the image: "))

k = int(input("Enter value of k: "))

db = client.MWD_Phase_1

feature_descriptors = list(db.feature_descriptors.find({}))
n = len(feature_descriptors)

euclidean_distances = []

for i in range(n):
  if i == image_id:
    continue

  euclidean_distances.append({"id": feature_descriptors[i]["_id"],  "similarity": math.dist(feature_descriptors[image_id]["layer3"], feature_descriptors[i]["layer3"])})

euclidean_similarity = sorted(euclidean_distances, key=lambda x: x["similarity"], reverse=False)

plt.imshow(caltectDataset[image_id].permute(1, 2, 0))
plt.show()

for i in range(k):
  print(euclidean_similarity[i])
  plt.imshow(caltectDataset[euclidean_similarity[i]["id"]].permute(1, 2, 0))
  plt.show()