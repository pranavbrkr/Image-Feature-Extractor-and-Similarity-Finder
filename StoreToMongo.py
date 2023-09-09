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

def storeInMongo(image_id):
  data = {
    "_id": image_id,
    "color_moments": extractCM10x10(caltectDataset, image_id),
    "hog": extractHOG(caltectDataset, image_id),
    "avgpool": extractResnetAvgpool1024(caltectDataset, image_id),
    "layer3": extractResnetLayer3(caltectDataset, image_id),
    "fc": extractResnetFc(caltectDataset, image_id),
  }

  if feature_descriptors.find_one({"_id": image_id}):
    feature_descriptors.update_one({"_id": image_id}, {"$set": data})
    print(f"Updated feature descriptors for Image {image_id}")
  else:
    feature_descriptors.insert_one(data)
    print(f"Inserted feature descriptors for Image {image_id}")
 
n = len(caltectDataset)
for image_id in range(n//2):
  storeInMongo(image_id)
# with parallel_config(backend='threading', n_jobs=5):
#   Parallel()(delayed(storeInMongo)(i) for i in range(300))

print(f"{(time.time() - start_time)} seconds")