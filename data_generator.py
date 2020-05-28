import numpy as np
import numbers
from scipy.special import softmax
from sklearn.datasets import make_classification

SEED = 43

class SyntheticDataGenerator:

    def __init__(self,
        seed =43,
        n_features = 2,
        n_classes = 2,
        n_clusters_per_class = 2,
        ):
        self.seed = seed
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_clusters_per_class = n_clusters_per_class
        

    def make_dataset(self, n_samples, weights, sep):
        x, y = make_classification(n_samples=n_samples, n_features=self.n_features, n_informative=self.n_features,
                        n_redundant=0, n_repeated=0, n_classes=self.n_classes,
                        n_clusters_per_class=2, weights=weights, flip_y=0.0,
                        class_sep=sep, hypercube=True, shift=0.0, scale=1.0,
                        shuffle=True, random_state=None)

        
        return {'x': x, 'y': y}
