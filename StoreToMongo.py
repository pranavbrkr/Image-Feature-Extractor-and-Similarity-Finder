from ExtractFeatureDescriptors import *
from pymongo import MongoClient
import time
from joblib import Parallel, parallel_config, delayed

start_time = time.time()

client = MongoClient()
client = MongoClient(host="localhost", port=27017)

caltectDataset = loadDataset()

db = client.MWD_Phase_1

feature_descriptors = db.feature_descriptors

n = len(caltectDataset)

count = 0

f = open('data.json', 'wb')
jsonarray = []

for image_id in range(n):
  if checkChannel(caltectDataset[image_id][1]):
    data = {
      "_id": image_id,
      "color_moments": extractCM10x10(caltectDataset[image_id][1]),
      "hog": extractHOG(caltectDataset[image_id][1]),
      "avgpool": extractResnetAvgpool1024(caltectDataset[image_id][1]),
      "layer3": extractResnetLayer3(caltectDataset[image_id][1]),
      "fc": extractResnetFc(caltectDataset[image_id][1]),
    }

  if feature_descriptors.find_one({"_id": image_id}):
    feature_descriptors.update_one({"_id": image_id}, {"$set": data})
    print(f"Updated feature descriptors for Image {image_id}")
  else:
    feature_descriptors.insert_one(data)
    print(f"Inserted feature descriptors for Image {image_id}")
 
print(f"{(time.time() - start_time)} seconds")