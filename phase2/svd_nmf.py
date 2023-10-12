import numpy as np

def svd(matrix, k, tolerance=1e-6):
    # Step 1: Compute the covariance matrix
    cov_matrix = np.dot(matrix.T, matrix)

    # Step 2: Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Step 3: Sort the eigenvalues and corresponding eigenvectors
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    # Step 4: Compute the singular values and the left and right singular vectors
    singular_values = np.sqrt(eigenvalues)
    left_singular_vectors = np.dot(matrix, eigenvectors)
    right_singular_vectors = eigenvectors

    # Step 5: Normalize the singular vectors
    for i in range(left_singular_vectors.shape[1]):
        left_singular_vectors[:, i] /= singular_values[i]

    for i in range(right_singular_vectors.shape[1]):
        right_singular_vectors[:, i] /= singular_values[i]

    # Keep only the top k singular values and their corresponding vectors
    singular_values = singular_values[:k]
    left_singular_vectors = left_singular_vectors[:, :k]
    right_singular_vectors = right_singular_vectors[:, :k]

    return left_singular_vectors, np.diag(singular_values), right_singular_vectors.T
import numpy as np


def nmf(matrix, k, num_iterations=100):
    d1, d2 = matrix.shape
    # Initialize W and H matrices with random non-negative values
    W = np.random.rand(d1, k)
    H = np.random.rand(k, d2)

    for iteration in range(num_iterations):
        # Update H matrix
        numerator_h = np.dot(W.T, matrix)
        denominator_h = np.dot(np.dot(W.T, W), H)
        H *= numerator_h / denominator_h

        # Update W matrix
        numerator_w = np.dot(matrix, H.T)
        denominator_w = np.dot(W, np.dot(H, H.T))
        W *= numerator_w / denominator_w

    return W, H

# Example usage
if __name__ == "__main__":
    # Create a sample matrix
    A = np.random.rand(1000,400)

    # Specify the desired reduced dimension (k)
    k = 10

    # Compute the SVD
    U, S, VT = svd(A, k)

    print("SVD")
    print("U (d1 x k):")
    print(U)
    print(U.shape)

    print("S (k x k):")
    print(S)
    print(S.shape)

    print("Vt (k x d2):")
    print(VT)
    print(VT.shape)

    # Compute the NMF
    W, H = nmf(A, k)
    print("NMF")
    print("Original Matrix:")
    print(A)
    print(A.shape)
    print("Reduced Matrix (W):")
    print(W)
    print(W.shape)

