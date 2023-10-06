import json
from pymongo import MongoClient
from task0a import *
import scipy
import numpy as np
from sklearn.decomposition import NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans


client = MongoClient()
client = MongoClient(host="localhost", port=27017)

# Select the database
db = client.Multimedia_Web_DBs

# caltechDataset = loadDataset()

# Fetch all documents from the collection and then sort them by "_id"
feature_descriptors = list(db.Caltech101_Feature_Descriptors.find({}))
feature_descriptors = sorted(list(db.Caltech101_Feature_Descriptors.find({})), key=lambda x: x["_id"], reverse=False)

num_labels = 101

def extractKLatentSemantics(k, feature_model, dim_reduction):

  feature_vectors = [x[feature_model] for x in feature_descriptors if x["_id"] % 2 == 0]
  feature_labels = [x["label"] for x in feature_descriptors if x["_id"] % 2 == 0]
  feature_ids = [x["_id"] for x in feature_descriptors if x["_id"] % 2 == 0]

  filename = ''


  match dim_reduction:

    case 1:
      filename = f'{feature_model}-svd-semantics.json'
      U, S, Vh = scipy.sparse.linalg.svds(np.array(feature_vectors), k=k)
      k_latent_semantics = sorted(list(zip(feature_ids, U.tolist())), key = lambda x: x[1][0], reverse = True)

    case 2:
      filename = f'{feature_model}-nnmf-semantics.json'
      model = NMF(n_components = k, init = 'random', solver = 'cd', alpha_H = 0.01, alpha_W = 0.01, max_iter = 10000)
      min_value = np.min(feature_vectors)
      feature_vectors_shifted = feature_vectors - min_value
      U = model.fit_transform(np.array(feature_vectors_shifted))
      k_latent_semantics = sorted(list(zip(feature_ids, U.tolist())), key = lambda x: x[1][0], reverse = True)

    case 3:
      filename = f'{feature_model}-lda-semantics.json'
      U = LinearDiscriminantAnalysis(n_components = k).fit_transform(feature_vectors, feature_labels)
      k_latent_semantics = sorted(list(zip(feature_ids, U.tolist())), key = lambda x: x[1][0], reverse = True)

    case 4:
      filename = f'{feature_model}-kmeans-semantics.json'
      kmeans = KMeans(n_clusters = k)
      kmeans.fit(feature_vectors)
      U = kmeans.transform(feature_vectors)
      k_latent_semantics = sorted(list(zip(feature_ids, U.tolist())), key = lambda x: x[1][0], reverse = True)
  
  k_latent_semantics = [{"_id": item[0], "semantics": item[1]} for item in k_latent_semantics]
  with open(filename, 'w', encoding='utf-8') as f:
    json.dump(k_latent_semantics, f, ensure_ascii = False)

def main():

  # Load dataset

  # User input for Image ID
  k = int(input("Enter k: "))

  features = ['color_moments', 'hog', 'layer3', 'avgpool', 'fc']

  # User input for feature model to extract
  print("\n1: Color moments")
  print("2: HOG")
  print("3: Resnet50 Avgpool layer")
  print("4: Resnet50 Layer 3")
  print("5: Resnet50 FC layer")
  feature_model = features[int(input("Select the feature model: ")) - 1]

  print("\n1. SVD")
  print("2. NNMF")
  print("3. LDA")
  print("4. k-means")
  dim_reduction = int(input("Select the dimensionality reduction technique: "))

  extractKLatentSemantics(k, feature_model, dim_reduction)

  

if __name__ == "__main__":
   main()