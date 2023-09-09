from ExtractFeatureDescriptors import *
from pymongo import MongoClient

client = MongoClient()
client = MongoClient(host="localhost", port=27017)

caltectDataset = loadDataset()

db = client.MWD_Phase_1

testcoll = db.feature_descriptors

for i in range(10):
  data = {
    "_id": i,
    "color_moments": extractCM10x10(caltectDataset, i).tolist(),
    "hog": extractHOG(caltectDataset, i).tolist(),
    "avgpool": extractResnetAvgpool1024(caltectDataset, i).tolist(),
    "layer3": extractResnetLayer3(caltectDataset, i).tolist(),
    "fc": extractResnetFc(caltectDataset, i).tolist(),
  }

  result = testcoll.insert_one(data)

  print(result)