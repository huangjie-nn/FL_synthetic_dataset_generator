import numpy as np
import numbers
import math
from scipy.special import softmax
from collections import Counter


NUM_DIM = 10

class SyntheticDataset:

    def __init__(
            self,
            num_classes=2,
            seed=931231,
            num_dim=NUM_DIM,
            prob_clusters=[1.0],
            n_parties = 2,
            x_level_noise = 1):

    
        np.random.seed(seed)
        self.num_classes = num_classes
        self.num_dim = num_dim
        self.num_clusters = len(prob_clusters)
        self.prob_clusters = prob_clusters
        self.side_info_dim = self.num_clusters
        self.n_parties = n_parties
        self.x_level_noise = x_level_noise
        self.Q = np.random.normal(
            loc=0.0, scale=1.0, size=(self.num_dim + 1, self.num_classes, self.side_info_dim))

        self.Sigma = np.zeros((self.num_dim, self.num_dim))
        for i in range(self.num_dim):
            self.Sigma[i, i] = (i + 1)**(-1.2)

        self.means = self._generate_clusters()


    def get_tasks(self, num_samples, weights, noises, loc_list):

        datasets = {}


        cluster_idx = np.random.choice(
            range(self.num_clusters), size=None, replace=True, p=self.prob_clusters)

        for i, (s, w, n, loc) in enumerate(zip(num_samples, weights, noises, loc_list)):
            print('current party is number. %s' %i)
            new_task = self._generate_task(self.means[cluster_idx], cluster_idx, s, n, loc, w)
            datasets[str(i)] = new_task
        return datasets

    def _generate_clusters(self):
        means = []
        for i in range(self.num_clusters):
            loc = np.random.normal(loc=0, scale=1., size=None)
            mu = np.random.normal(loc=loc, scale=1., size=self.side_info_dim)
            means.append(mu)
        return means

    def _generate_x(self, loc, num_samples):
        # loc = np.random.normal(loc=B, scale=1.0, size=self.num_dim)
        samples = np.ones((num_samples, self.num_dim + 1))
        samples[:, 1:] = np.random.multivariate_normal(
            mean=loc, cov=self.Sigma, size=num_samples)

        return samples

    def _generate_y(self, x, model_info, noise):


        if self.x_level_noise == 1:
            x = x + np.random.normal(loc=0, scale=0.1, size=x.shape) * noise
        
        w = np.matmul(self.Q, model_info)
        
        num_samples = x.shape[0]



        prob = softmax(np.matmul(x, w))
        if self.x_level_noise == 0:

            prob = prob + noise * np.random.normal(loc=0., scale=0.1, size=(num_samples, self.num_classes), axis=1)
                
        y = np.argmax(prob, axis=1)
        return y, w, model_info

    def _generate_task(self, cluster_mean, cluster_id, num_samples, noise, loc, label_weights):

        print('in _generate_task')
        num_labels = [math.floor(num_samples * i) for i in label_weights]
        
        left_over = num_samples - sum(num_labels)
        num_labels[-1] += left_over

        model_info = np.random.normal(loc=cluster_mean, scale=0.1, size=cluster_mean.shape)
        

        x, y=self._get_sets(num_labels, model_info, num_samples, noise, loc, [], [], 0)

        print('expected label distribution is %s' %num_labels)
        sorted_counts = self._count_helper(y, num_labels)

        print('current potion after populating step %s' %sorted_counts)

        final_x, final_y = self._trim(x, y, num_labels)

        sorted_counts = self._count_helper(final_y, num_labels)

        print('current potion after trimming step %s' %sorted_counts)

        return {'x': final_x, 'y': final_y}

    def _count_helper(self, y, num_labels):

        counts = Counter(y)

        for i in range(len(num_labels)):
            value = counts.get(i,None)
            if value is None:
                counts[i]=0

        sorted_counts = [counts[key] for key in sorted(counts.keys(), reverse=False)]

        return sorted_counts

    def _trim(self, x, y, num_labels):
        x = np.array(x)
        y = np.array(y)
        idx_to_keep = []
        for i in range(len(num_labels)):
            indexes = [idx for idx,label in enumerate(y) if label == i]
            if len(indexes) > num_labels[i]:
                idx_to_keep.extend(indexes[:num_labels[i]])

        final_x = x[idx_to_keep]
        final_y = y[idx_to_keep]
        return final_x, final_y


    def _get_sets(self, num_labels, model_info, num_samples, noise, loc, full_x=[], full_y=[], count = 0):

        x, y = self._generate_trialset(model_info, num_samples, noise, loc)
        full_x.extend(x)
        full_y.extend(y) 
        count +=1
        sorted_counts = self._count_helper(full_y, num_labels)
        for expect, reality in zip(num_labels, sorted_counts):
            if expect > reality:
                x, y = self._get_sets(num_labels, model_info, num_samples, noise, loc, full_x, full_y, count)
        return full_x, full_y



    def _generate_trialset(self, model_info, num_samples, noise, loc):
        x = self._generate_x(loc, num_samples)
        y, w, model_info = self._generate_y(x, model_info, noise)
        x = x[:, 1:]

        return x, y