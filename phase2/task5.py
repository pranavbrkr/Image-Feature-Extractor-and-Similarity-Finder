import json
import math
from pymongo import MongoClient
import scipy
import svd_nmf as svd_nmf
import numpy as np
from sklearn.decomposition import NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans

client = MongoClient()
client = MongoClient(host = "localhost", port = 27017)

# Select the database
db = client.Multimedia_Web_DBs

# Fetch all documents from the collection and then sort them by "_id"
feature_descriptors = list(db.Caltech101_Feature_Descriptors.find({}))
feature_descriptors = sorted(list(db.Caltech101_Feature_Descriptors.find({})), key=lambda x: x["_id"], reverse=False)

def extractKLatentSemantics(k, image_sim_matrix, feature_model, dim_reduction):

  feature_ids = [x["_id"] for x in feature_descriptors if x["_id"] % 2 == 0]
  feature_labels = [x["label"] for x in feature_descriptors if x["_id"] % 2 == 0]

  filename = 'ls3.json'

  match dim_reduction:

    case 1:
      U, S, Vh = svd_nmf.svd(matrix = np.array(image_sim_matrix), k = k)
      print(U)
      print(U.shape)
      print(S)
      print(S.shape)
      print(Vh)
      print(Vh.shape)
      k_latent_semantics = sorted(list(zip(feature_ids, U.tolist())), key = lambda x: x[1][0], reverse = True)

    case 2:
      min_value = np.min(image_sim_matrix)
      feature_vectors_shifted = image_sim_matrix - min_value
      W, H = svd_nmf.nmf(matrix = np.array(feature_vectors_shifted), k = k)
      print(W)
      print(W.shape)
      print(H)
      print(H.shape)
      k_latent_semantics = sorted(list(zip(feature_ids, W.tolist())), key = lambda x: x[1][0], reverse = True)

    case 3:
      U = LinearDiscriminantAnalysis(n_components = k).fit_transform(image_sim_matrix, feature_labels)
      k_latent_semantics = sorted(list(zip(feature_ids, U.tolist())), key = lambda x: x[1][0], reverse = True)

    case 4:
      kmeans = KMeans(n_clusters = k)
      kmeans.fit(image_sim_matrix)
      U = kmeans.transform(image_sim_matrix)
      k_latent_semantics = sorted(list(zip(feature_ids, U.tolist())), key = lambda x: x[1][0], reverse = True)
  
  k_latent_semantics = [{"_id": item[0], "semantics": item[1]} for item in k_latent_semantics]
  with open(filename, 'w', encoding='utf-8') as f:
    json.dump(k_latent_semantics, f, ensure_ascii = False)

def findLabelLabelSimMatrix(feature_model):

  label_sim_matrix = []
  label_mean_vectors = []

  for label in range(101):
    label_vectors = [x[feature_model] for x in feature_descriptors if x["label"] == label and x["_id"] % 2 == 0]
    mean_vector = [sum(col)/len(col) for col in zip(*label_vectors)]
    label_mean_vectors.append(mean_vector)
  

  n = len(label_mean_vectors)

  label_sim_matrix = np.zeros((n, n))

  for i in range(n):
    for j in range(i + 1, n):

      match feature_model:

        case "color_moments":
          label_sim_matrix[i][j] = label_sim_matrix[j][i] = math.dist(label_mean_vectors[i], label_mean_vectors[j])
        
        case "hog":
          label_sim_matrix[i][j] = label_sim_matrix[j][i] = (np.dot(label_mean_vectors[i], label_mean_vectors[j]) / (np.linalg.norm(label_mean_vectors[i]) * np.linalg.norm(label_mean_vectors[j])))

        case "avgpool" | "layer3" | "fc":
          label_sim_matrix[i][j] = label_sim_matrix[j][i] = scipy.stats.pearsonr(label_mean_vectors[i], label_mean_vectors[j]).statistic
          
  return label_sim_matrix


def main():

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

  label_sim_matrix = findLabelLabelSimMatrix(feature_model)

  extractKLatentSemantics(k, label_sim_matrix, feature_model, dim_reduction)

  

if __name__ == "__main__":
   main()