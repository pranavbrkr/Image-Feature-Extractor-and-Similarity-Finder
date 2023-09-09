from ExtractFeatureDescriptors import *
from pymongo import MongoClient

client = MongoClient()
client = MongoClient(host="localhost", port=27017)

caltectDataset = loadDataset()

db = client.MWD_Phase_1

feature_descriptors = db.feature_descriptors

for i in range(10):
  data = {
    "_id": i,
    "color_moments": extractCM10x10(caltectDataset, i),
    "hog": extractHOG(caltectDataset, i),
    "avgpool": extractResnetAvgpool1024(caltectDataset, i),
    "layer3": extractResnetLayer3(caltectDataset, i),
    "fc": extractResnetFc(caltectDataset, i),
  }

  result = feature_descriptors.insert_one(data)

  print(result)