import numpy as np
import numbers
import math
from scipy.special import softmax
from collections import Counter


NUM_DIM = 10
GET_SETS_TOLERANCE = 1000

class SyntheticDataset:
    def __init__(self, seed, num_classes=2, num_dim=NUM_DIM, n_parties = 2,x_level_noise = 1):
        """
        Initialize synthetic dataset instance. 
        Args:
            num_classes (int):
                Int value indicating the number of classes in the current simulated fl task
            seed (int):
                Random seed for the synthetic dataset generator.
            num_dim (int):
                Int value indicating the number of features, i.e. the number of dimensions of the feature space.
            n_party (int):
                Number of parties for the current simulated fl task.
            x_level_noise (int):
                Default at 1. 
                Currently take in 1 and 0:
                if x_level_noise == 1, the label will be generated after noises are added into the features.
                if x_level_noise == 0, the label will be generated before noises are added into the features. 
        Returns:
             
        """
        np.random.seed(seed)
        self.total_attempts = 0
        self.seed = seed
        self.num_classes = num_classes
        self.num_dim = num_dim
        self.n_parties = n_parties
        self.x_level_noise = x_level_noise

        ########
        # self.Q is the global model. 
        # local model will be the addition of globla model and a perturbation 
        # matrix with small values in every entry with the same shape
        ########
        self.Q = np.random.normal(
            loc=0.0, scale=1.0, size=(self.num_dim + 1, self.num_classes))

        ########
        # self.Sigma is the covariance matrix for all parties for the x generation process, where 
        # the features are sampled from a multivariant Guassian distribution with mean defined in loc and 
        # covariance matrix of self.Sigma
        ########
        self.Sigma = np.zeros((self.num_dim, self.num_dim))
        for i in range(self.num_dim):
            self.Sigma[i, i] = (i + 1)**(-1.2)

    def get_tasks(self, num_samples, weights, noises, loc_list, perturb_list):
        """
        Get datasets for every party. Get global and local model weights. 
        Args:
            num_samples (list(int)):
                A list of int, indicating the total number of samples for each party.
            weights (dict):
                The label portion for each class of each party, defined by user.
            noise (list(float)):
                A list of float, indicating the amount of noises for each party.
            loc_list (np.array(num_parties, num_features)):
                Numpy array of loc for each feature of each party. 
                Row i indicate the loc for every feature for party i.
                Column j indicate the loc for every party for feature j.
                loc will serve as a mean for every feature of each party for the multivariant Gaussian sampling process.
        Returns:
            Datasets for all parties.
            Global and model weights. 
        """
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
        """
        Get features for a single party  
        Args:
            loc (list(float)):
                List of floats with length of num_features. Each represent the mean for that specific feature of which the 
                multivariant Gaussian sampling will be done.  
            num_samples (int):
                Number of data samples for this single party
        Returns:
            Generated features. 
        """
        samples = np.ones((num_samples, self.num_dim + 1))
        samples[:, 1:] = np.random.multivariate_normal(
            mean=loc, cov=self.Sigma, size=num_samples)

        return samples

    def _generate_y(self, x, noise, w):
        """
        Get labels for a single party, given the modle weights, features, and noises 
        NOTE: Each party's weight is the global model added a perturbation matrix. 
        Args:
            x (np.array(num_samples, num_dim +1)):
                Values for the feature for this party. 
            noise (float):
                A scalar which will scale the noise generated. If 0, there will be no noise apply to this party
            w (np.array(num_dim+1, num_classes)):
        Returns:
            Generated weights. 
        """
        if self.x_level_noise == 1:
            x = x + (np.random.normal(loc=0, scale=0.1, size=x.shape) * noise)
            num_samples = x.shape[0]
            prob = softmax(np.matmul(x, w))
            y = np.argmax(prob, axis=1)
        elif self.x_level_noise == 0:
            num_samples = x.shape[0]
            prob = softmax(np.matmul(x, w))
            y = np.argmax(prob, axis=1)
            x = x + (np.random.normal(loc=0, scale=0.1, size=x.shape) * noise)
        return y

    def _generate_task(self, num_samples, noise, loc, label_weights, perturb):
        """
        Get datasets for each party
        Args:
            num_samples (int):
                Number of data samples for this current party
            noise (float):
                Scale of the noise for this party
            loc (list(float)):
                List of float, each value indicate the mean for that feature. 
            label_weigths (list(float)):
                List of float, indicating the distribution of each label in this current party.
            perturb (np.array(num_dim + 1, num_classes)):
                The perturbation matrix, each entry with small value. 
                The perturb will be added into the global model to produce the local model for current party.
        Returns:
            Generated datasets. 
            Local model
        """
        print('in _generate_task')
        num_labels = [math.floor(num_samples * i) for i in label_weights]
        left_over = num_samples - sum(num_labels)
        num_labels[-1] += left_over
        local_Q = self.Q + perturb
        x, y = self._get_sets(num_labels, num_samples, noise, loc, local_Q,[], [], 0)
        print('expected label distribution is %s' %num_labels)
        sorted_counts = self._count_helper(y, num_labels)

        print('current portionafter populating step %s' %sorted_counts)

        final_x, final_y = self._trim(x, y, num_labels)

        sorted_counts = self._count_helper(final_y, num_labels)

        print('current portionafter trimming step %s' %sorted_counts)
        print(self.total_attempts)
        print(self.seed)
        return {'x': final_x, 'y': final_y}, local_Q

    def _count_helper(self, y, num_labels):
        """
        Count the occurance of each label in the given y list. 
        Args:
            y (list(int)):
                List of labels (numeric), indicating the label for each data sample
            num_labels (int):
                Number of classes
        Returns:
            A list, with index meaning the label value, and the value meaning the occurance.
        """
        counts = Counter(y)

        for i in range(len(num_labels)):
            value = counts.get(i,None)
            if value is None:
                counts[i]=0

        sorted_counts = [counts[key] for key in sorted(counts.keys(), reverse=False)]

        return sorted_counts

    def _trim(self, x, y, num_labels):
        """
        Helper function to trim the extra data samples  
        Args:
            x (np.array(num_samples, num_dim+1)):
                Generated feature values
            y (np.array(num_samples)):
                All the labels
            num_labels (list(int)):
                A list of int, index indicates the label value, value indicates the desired occurance.
        Returns:
            Trimmed x and y
        """
        x = np.array(x)
        y = np.array(y)
        idx_to_keep = []
        for i in range(len(num_labels)):
            counter = 0
            for j in range(len(y)):
                if y[j] == i:
                    if counter < num_labels[i]:
                        idx_to_keep.append(j)
                        counter += 1
        final_x = x[idx_to_keep]
        final_y = y[idx_to_keep]
        return final_x, final_y

    def _get_sets(self, num_labels, num_samples, noise, loc, w, full_x=[], full_y=[], count = 0):
        """
        Helper function to generate datasets for each party, that ideally should fulfill the label distribution indicated by user.
        NOTE: the function will be in a infinite loop until the desired label distribution reached, or until maximum tolerant number 
        of trial is ahieved.  
        Args:
            num_labels (list(int)):
                A list of int, index indicates the label value, value indicates the desired occurance.
            num_samples (int):
                Total number of samples for this current party
            noise (float):
                Noise for the specific party
            loc (list(float)):
                List of float, each value is the mean for the specific feature for the multi-variant Guassian sampling 
                when generating x. 
            w (np.array(num_dim+1, num_classes)):
                Local model weights, obtained by adding a perturbation matrix with small values to the global model.
            full_x (list([])):
                A full list of x features, for collecting all generated samples before trimming for this current party. 
            full_y (list[[]]):
                A full list of y label, for collecting all generated sample before trimming for this current party.
            count (int):
                Start at 0. Count the nubmer of recursive loop. 
        Returns:
            Full set of x and y
        """ 
        x, y = self._generate_trialset(num_samples, noise, loc, w)
        full_x.extend(x)
        full_y.extend(y)
        sorted_counts = self._count_helper(full_y, num_labels)
        self.total_attempts += 1
        for class_idx in range(len(num_labels)):
            required_amount = num_labels[class_idx]
            generated_amount = sorted_counts[class_idx]
            attempt_count = 0
            while required_amount > generated_amount:
                x, y = self._generate_trialset(num_samples, noise, loc, w)
                for sample_idx in range(len(y)):
                    if y[sample_idx] == class_idx:
                        full_x.extend([x[sample_idx]])
                        full_y.extend([y[sample_idx]])
                generated_amount = self._count_helper(full_y, num_labels)[class_idx]
                attempt_count += 1
                self.total_attempts += 1
                if attempt_count > GET_SETS_TOLERANCE:
                    break
        return full_x, full_y

    def _generate_trialset(self,num_samples, noise, loc, w):
        """
        A single round of generating x and y 
        Args:
            num_samples (int):
                Total number of samples for this current party
            noise (float):
                Noise for the specific party
            loc (list(float)):
                List of float, each value is the mean for the specific feature for the multi-variant Guassian sampling 
                when generating x. 
            w (np.array(num_dim+1, num_classes)):
                Local model weights, obtained by adding a perturbation matrix with small values to the global model.
        Returns:
            x and y
        """        
        x = self._generate_x(loc, num_samples)
        y = self._generate_y(x, noise, w)
        x = x[:, 1:]

        return x, y
