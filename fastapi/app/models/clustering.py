import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BaseClusterer:
    def cluster(self, features, allow_noise=False):
        raise NotImplementedError
    
    def assign_noise(self, labels):
        labels = np.array(labels, copy=True)
        mask = labels == -1
        start = labels.max() + 1
        labels[mask] = np.arange(start, start + mask.sum())
        return labels

class HDBSCANClusterer(BaseClusterer):
    def __init__(self, min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0.1):
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon
        )

    def cluster(self, features, allow_noise=False):
        labels = self.clusterer.fit_predict(np.array(features, copy=True))
        if not allow_noise: 
            labels = self.assign_noise(labels)
        return labels

# Use this for small samples
class CosineClusterer(BaseClusterer):
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def cluster(self, features, allow_noise=False):
        features = np.array(features, copy=True)
        n_samples = len(features)
        if n_samples == 0:
            return np.array([], dtype=int)
        elif n_samples == 1:
            return np.array([0], dtype=int)
        
        labels = np.full(n_samples, -1, dtype=int)
        sim = cosine_similarity(features)
        current_label = 0
        for i in range(n_samples):
            if labels[i] == -1:
                mask = (labels == -1) & (sim[i] >= self.threshold)
                labels[mask] = current_label
                current_label += 1
        if not allow_noise:
            labels = self.assign_noise(labels)
        return labels