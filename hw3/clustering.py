
import numpy as np


class Cluster:
    def __init__(self):
        pass

    # PCA
    def pca(self, tfidf_matrix, n_components):
        tfidf_array = tfidf_matrix.to_numpy()
        mean = np.mean(tfidf_array, axis=0)
        centered_matrix = tfidf_array - mean
        covariance_matrix = np.cov(centered_matrix, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        selected_eigenvectors = sorted_eigenvectors[:, :n_components]

        reduced_matrix = np.dot(centered_matrix, selected_eigenvectors)
        reduced_matrix = np.real(reduced_matrix)
        return reduced_matrix

    @staticmethod
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    @staticmethod
    def cosine_distance(point1, point2):
        dot_product = np.dot(point1, point2)
        norm1 = np.linalg.norm(point1)
        norm2 = np.linalg.norm(point2)
        return 1.0 - (dot_product / (norm1 * norm2))

    def kmeans(self, reduced_matrix, k, distance_metric='euclidean', max_iterations=100):
        # Initialize centroids randomly
        centroids = reduced_matrix[np.random.choice(len(reduced_matrix), k, replace=False)]
        n_points = len(reduced_matrix)

        for _ in range(max_iterations):
            # Assign data points to the nearest centroid based on the chosen distance metric
            if distance_metric == 'euclidean':
                distance_function = self.euclidean_distance
            elif distance_metric == 'cosine':
                distance_function = self.cosine_distance
            else:
                raise ValueError("Invalid distance metric. Use 'euclidean' or 'cosine'.")

            labels = np.array(
                [np.argmin([distance_function(point, centroid) for centroid in centroids]) for point in reduced_matrix])

            # Update centroids
            new_centroids = [reduced_matrix[labels == i].mean(axis=0) for i in range(k)]
            new_centroids = np.array(new_centroids)

            # Check for convergence
            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids
        return centroids, labels

    def confusion_matrix(self, true_labels, predicted_labels):
        confusion_matrix = np.zeros((len(np.unique(true_labels)), len(np.unique(predicted_labels))))

        for cluster_id in np.unique(predicted_labels):
            mask = (predicted_labels == cluster_id)
            cluster_label = np.argmax(np.bincount(true_labels[mask]))
            confusion_matrix[cluster_label, cluster_id] = np.sum(mask)

        precision = np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)
        recall = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return confusion_matrix, precision, recall, f1_score








