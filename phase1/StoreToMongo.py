from ExtractFeatureDescriptors import *
from pymongo import MongoClient


#
#
# Connect to the MongoDB Client
client = MongoClient()
client = MongoClient(host="localhost", port=27017)

# Load dataset
caltectDataset = loadDataset()

# Selecct database
db = client.Multimedia_Web_DBs

# select collection
feature_descriptors = db.Caltech101_Feature_Descriptors

n = len(caltectDataset)

for image_id in range(n):
  if checkChannel(caltectDataset[image_id][1]):
  # Calculate all feature descriptors if image has 3 channels
    data = {
      "_id": image_id,
      "color_moments": extractCM10x10(caltectDataset[image_id][1]),
      "hog": extractHOG(caltectDataset[image_id][1]),
      "avgpool": extractResnetAvgpool1024(caltectDataset[image_id][1]),
      "layer3": extractResnetLayer3(caltectDataset[image_id][1]),
      "fc": extractResnetFc(caltectDataset[image_id][1]),
      "label": caltectDataset[image_id][2],
    }
  else:
    # Else calculate just the HOG
    gray_to_rgb = torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    rgb_image = gray_to_rgb(caltectDataset[image_id][1])

    data = {
      "_id": image_id,
      "color_moments": extractCM10x10(rgb_image),
      "hog": extractHOG(rgb_image),
      "avgpool": extractResnetAvgpool1024(rgb_image),
      "layer3": extractResnetLayer3(rgb_image),
      "fc": extractResnetFc(rgb_image),
      "label": caltectDataset[image_id][2],
    }

  if feature_descriptors.find_one({"_id": image_id}):
    # If image ID exists, just update the document
    feature_descriptors.update_one({"_id": image_id}, {"$set": data})
    print(f"Updated feature descriptors for Image {image_id}")
  else:
    # Else just insert new document
    feature_descriptors.insert_one(data)
    print(f"Inserted feature descriptors for Image {image_id}")