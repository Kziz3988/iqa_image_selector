import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

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
    def __init__(self, min_cluster_size=2):
        def cosine_distance(u, v):
            return 1 - np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=cosine_distance
        )

    def cluster(self, features, allow_noise=False):
        labels = self.clusterer.fit_predict(np.array(features, copy=True))
        if not allow_noise: 
            labels = self.assign_noise(labels)
        return labels

# Use this for small samples
class AgglomerativeClusterer(BaseClusterer):
    def __init__(self):
        self.clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            distance_threshold=0.2,
            linkage='average'
        )

    def cluster(self, features, allow_noise=False):
        labels = self.clusterer.fit_predict(np.array(features, copy=True))
        if not allow_noise: 
            labels = self.assign_noise(labels)
        return labels

class ClustererFactory:
    @staticmethod
    def get_model(name='HDBSCAN'):
        if name == 'HDBSCAN':
            return HDBSCANClusterer()
        elif name == 'Agglomerative':
            return AgglomerativeClusterer()
        else:
            raise ValueError(f"Unknown clustering method: {name}")