import numpy as np


# this class add noise to data
class Noise:
    def __init__(self):
        pass

    # this function add laplace noise to some data, real data is object_array. Sensitivity of
    # laplace mechanism is sensitivity1 and privacy budget is epsilon1.
    def add_laplace(self, object_array, epsilon1, sensitivity1, if_regularize=True):
        lambda1 = sensitivity1 / epsilon1
        shape1 = object_array.shape
        noisy_object = object_array + np.random.laplace(scale=lambda1, size=shape1)
        if if_regularize:
            noisy_object = self.positive_regulation(noisy_object)
        return noisy_object

    # this function change meaningless negative value to zero. These negative value comes from
    # noise adding and usually meaningless such as density. This function sum up all negative value
    # and add this negative sum to the smallest positive value. Iter this until the negative sum runs out.
    def positive_regulation(self, noisy_array1):
        noisy_array1 = noisy_array1
        sort_indices = np.argsort(noisy_array1)
        sorted_array1 = noisy_array1[sort_indices]
        negative_indices = np.argwhere(sorted_array1 < 0).reshape(-1)
        positive_indices = np.argwhere(sorted_array1 > 0).reshape(-1)
        if negative_indices.size > 0:
            if positive_indices.size > 0:
                negative_sum = np.sum(sorted_array1[negative_indices])
                positive_sum = np.sum(sorted_array1[positive_indices])
                if positive_sum > - negative_sum:
                    index_larger_than_negative = 1
                    tem_poi = np.sum(sorted_array1[positive_indices[0:index_larger_than_negative]])
                    while - negative_sum > tem_poi:
                        index_larger_than_negative += 1
                        tem_poi = np.sum(sorted_array1[positive_indices[0:index_larger_than_negative]])
                    sorted_array1[positive_indices[index_larger_than_negative - 1]] = np.sum(
                        sorted_array1[positive_indices[0:index_larger_than_negative]]) + negative_sum
                    if index_larger_than_negative > 1:
                        sorted_array1[positive_indices[0:index_larger_than_negative - 1]] = 0
                    sorted_array1[negative_indices] = 0
                    indices_inverse = np.argsort(sort_indices)
                    array_after_regulation = sorted_array1[indices_inverse]
                else:
                    array_after_regulation = np.zeros(noisy_array1.size)
            else:
                array_after_regulation = np.zeros(noisy_array1.size)
        else:
            array_after_regulation = noisy_array1
        return array_after_regulation

    # this function regulate matrix after adding noise by row
    def positive_regulation_for_markov_matrix(self, noisy_matrix1, regulation_method='queue_minus'):
        if regulation_method == 'queue_minus':
            for rows in range(noisy_matrix1.shape[0]):
                noisy_matrix1[rows] = self.positive_regulation(noisy_matrix1[rows])
        elif regulation_method == 'truncation':
            noisy_matrix1[noisy_matrix1 < 0] = 0
        return noisy_matrix1
