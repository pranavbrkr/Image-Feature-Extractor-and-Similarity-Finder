from ExtractFeatureDescriptors import *
from pymongo import MongoClient
import torchvision

client = MongoClient()
client = MongoClient(host="localhost", port=27017)

caltectDataset = torchvision.datasets.Caltech101(root='./Dataset/',download=True)
print(len(caltectDataset))

db = client.test

testcoll = db.testcoll

for i in range(10):
  data = {
    "_id": i,
    "color_moments": extractCM10x10(i).tolist(),
    "hog": extractHOG(i).tolist(),
    "avgpool": extractResnetAvgpool1024(i).tolist(),
    "layer3": extractResnetLayer3(i).tolist(),
    "fc": extractResnetFc(i).tolist(),
  }

  result = testcoll.insert_one(data)

  print(result)