from ExtractFeatureDescriptors import *
from pymongo import MongoClient

client = MongoClient()
client = MongoClient(host="localhost", port=27017)

caltectDataset = loadDataset()

db = client.MWD_Phase_1

feature_descriptors = db.feature_descriptors

for image_id in range(10):
  data = {
    "_id": image_id,
    "color_moments": extractCM10x10(caltectDataset, image_id),
    "hog": extractHOG(caltectDataset, image_id),
    "avgpool": extractResnetAvgpool1024(caltectDataset, image_id),
    "layer3": extractResnetLayer3(caltectDataset, image_id),
    "fc": extractResnetFc(caltectDataset, image_id),
  }

  if feature_descriptors.find_one({"_id": image_id}):
    result = feature_descriptors.update_one({"_id": image_id}, {"$set": data})
  else:
    result = feature_descriptors.insert_one(data)

  print(result)