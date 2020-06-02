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
            n_parties = 2,
            x_level_noise = 1):

    
        np.random.seed(seed)
        self.num_classes = num_classes
        self.num_dim = num_dim
        self.n_parties = n_parties
        self.x_level_noise = x_level_noise
        self.Q = np.random.normal(
            loc=0.0, scale=1.0, size=(self.num_dim + 1, self.num_classes))

        self.Sigma = np.zeros((self.num_dim, self.num_dim))
        for i in range(self.num_dim):
            self.Sigma[i, i] = (i + 1)**(-1.2)



    def get_tasks(self, num_samples, weights, noises, loc_list, perturb_list):

        datasets = {}
        model_weight_dic = {}

        for i, (s, w, n, loc, perturb) in enumerate(zip(num_samples, weights, noises, loc_list, perturb_list)):
            print('current party is number. %s' %i)
            new_task, w = self._generate_task(s, n, loc, w, perturb)
            datasets[str(i)] = new_task
            model_weight_dic[str(i)] = w.tolist()
        model_weight_dic['global'] = self.Q.tolist()
        return datasets, model_weight_dic

    def _generate_x(self, loc, num_samples):
        # loc = np.random.normal(loc=B, scale=1.0, size=self.num_dim)
        samples = np.ones((num_samples, self.num_dim + 1))
        samples[:, 1:] = np.random.multivariate_normal(
            mean=loc, cov=self.Sigma, size=num_samples)

        return samples

    def _generate_y(self, x, noise, w):


        if self.x_level_noise == 1:
            x = x + np.random.normal(loc=0, scale=0.1, size=x.shape) * noise
        
        num_samples = x.shape[0]



        prob = softmax(np.matmul(x, w))
        if self.x_level_noise == 0:

            prob = prob + noise * np.random.normal(loc=0., scale=0.1, size=(num_samples, self.num_classes), axis=1)
                
        y = np.argmax(prob, axis=1)
        return y

    def _generate_task(self, num_samples, noise, loc, label_weights, perturb):

        print('in _generate_task')
        num_labels = [math.floor(num_samples * i) for i in label_weights]
        
        left_over = num_samples - sum(num_labels)
        num_labels[-1] += left_over

        local_Q = self.Q + perturb

        x, y=self._get_sets(num_labels, num_samples, noise, loc, local_Q,[], [], 0)

        print('expected label distribution is %s' %num_labels)
        sorted_counts = self._count_helper(y, num_labels)

        print('current potion after populating step %s' %sorted_counts)

        final_x, final_y = self._trim(x, y, num_labels)

        sorted_counts = self._count_helper(final_y, num_labels)

        print('current potion after trimming step %s' %sorted_counts)

        return {'x': final_x, 'y': final_y}, local_Q

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


    def _get_sets(self, num_labels, num_samples, noise, loc, w, full_x=[], full_y=[], count = 0):

        x, y = self._generate_trialset(num_samples, noise, loc, w)
        full_x.extend(x)
        full_y.extend(y) 
        count +=1
        sorted_counts = self._count_helper(full_y, num_labels)
        for expect, reality in zip(num_labels, sorted_counts):
            if expect > reality:
                x, y = self._get_sets(num_labels, num_samples, noise, loc, w, full_x, full_y, count)
        return full_x, full_y



    def _generate_trialset(self,num_samples, noise, loc, w):
        x = self._generate_x(loc, num_samples)
        y = self._generate_y(x, noise, w)
        x = x[:, 1:]

        return x, y