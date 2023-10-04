from pymongo import MongoClient

client = MongoClient()
client = MongoClient(host="localhost", port=27017)

# Select the database
db = client.Multimedia_Web_DBs

# Fetch all documents from the collection and then sort them by "_id"
feature_descriptors = list(db.Caltech101_Feature_Descriptors.find({}))
feature_descriptors = sorted(list(db.Caltech101_Feature_Descriptors.find({})), key=lambda x: x["_id"], reverse=False)

num_labels = 101

def calculate_label_means(l, feature_model):
  
  label_vectors = [x[feature_model] for x in feature_descriptors if x["label"] == l]
  num = len(label_vectors)
  length = (len(label_vectors[0]))
  label_mean_vectors = [sum(col)/len(col) for col in zip(*label_vectors)]

  print(len(label_mean_vectors))
  print(label_mean_vectors)


def main():

  # Load dataset

  # User input for Image ID
  l = int(input("Enter query label: "))
  image_id = int(input("Enter image ID: "))
  k = int(input("Enter k: "))

  features = ['color_moments', 'hog', 'layer3', 'avgpool', 'fc']

  # User input for feature model to extract
  print("1: Color moments")
  print("2: HOG")
  print("3: Resnet50 Avgpool layer")
  print("4: Resnet50 Layer 3")
  print("5: Resnet50 FC layer")
  feature_model = features[int(input("Enter which feature model to extract: ")) - 1]

  mean_vectors = calculate_label_means(l, feature_model)

if __name__ == "__main__":
   main()