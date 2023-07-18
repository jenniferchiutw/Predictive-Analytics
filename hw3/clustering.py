
import numpy as np


class Cluster:
    def __init__(self):
        pass

    # PCA
    def pca(self, tfidf_matrix, n_components):
        tfidf_array = tfidf_matrix.to_numpy()

        mean = np.mean(tfidf_array, axis=0)
        centered_X = tfidf_array - mean
        covariance_matrix = np.cov(centered_X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        selected_eigenvectors = sorted_eigenvectors[:, :n_components]
        reduced_matrix = np.dot(centered_X, selected_eigenvectors)
        reduced_matrix = np.real(reduced_matrix)
        return reduced_matrix

    # K-means
    def kmeans(self, reduced_matrix, k, max_iterations=100):
        centroids = reduced_matrix[np.random.choice(range(reduced_matrix.shape[0]), size=k, replace=False)]

        for _ in range(max_iterations):
            distances = np.linalg.norm(reduced_matrix[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([reduced_matrix[labels == i].mean(axis=0) for i in range(k)])
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        return labels, centroids
