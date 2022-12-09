import numpy as np
import config.folder_and_file_names as config
from config.parameter_carrier import ParameterCarrier


class Filter:

    def __init__(self, cc: ParameterCarrier):
        self.cc = cc
        self.sensitivity_indicator = np.array([])
        self.real_state_number = 0

        # this parameter is a threshold percentage of out degree for states.
        # state larger than that percentage will be selected.
        self.out_degree_percentage = 0.01
        self.end_percentage = 0.02
        self.begin_percentage = 0.02

        self.distribution_threshold = 5

    #
    def find_sensitive_state(self, markov_matrix):
        cc1 = self.cc
        self.real_state_number = markov_matrix.shape[0] - 2
        degree_amount_indicator = self.degree_amount_sensitivity(markov_matrix)
        degree_distribution_indicator = self.degree_distribution_sensitivity(markov_matrix)
        extremely_large_degree_indicator = self.extremely_large_out_degree_sensitivity(markov_matrix)
        begin_indicator = self.begin_sensitivity(markov_matrix)
        end_indicator = self.end_sensitivity(markov_matrix)
        final_indicator = np.logical_and(degree_amount_indicator, degree_distribution_indicator)
        final_indicator = np.logical_or(final_indicator, extremely_large_degree_indicator)
        final_indicator = np.logical_or(final_indicator, begin_indicator)
        final_indicator = np.logical_or(final_indicator, end_indicator)
        self.sensitivity_indicator = final_indicator
        return final_indicator

    #
    def end_sensitivity(self, matrix):
        end_distribution = matrix[:, -1]
        end_distribution = end_distribution[:-2]
        end_indicator = (end_distribution > np.sum(end_distribution) * self.end_percentage)
        return end_indicator

    #
    def begin_sensitivity(self, matrix):
        begin_distribution = matrix[-2, :]
        begin_distribution = begin_distribution[:-2]
        begin_indicator = (begin_distribution > np.sum(begin_distribution) * self.begin_percentage)
        return begin_indicator

    #
    def extremely_large_out_degree_sensitivity(self, matrix):
        cc1 = self.cc
        out_degrees = matrix.sum(axis=1)
        total_degree = np.sum(out_degrees)
        threshold = total_degree * 0.02
        indicator = (out_degrees > threshold)
        indicator = indicator[:self.real_state_number]
        return indicator

    #
    def degree_amount_sensitivity(self, matrix):
        matrix = matrix[0: -2, 0: -2]
        degree_amount_of_states = np.sum(matrix, axis=1)
        amount_threshold = self.real_state_number * 1.414 / (self.cc.total_epsilon * self.cc.epsilon_partition[1])
        indicator = (degree_amount_of_states > amount_threshold)
        indicator = indicator[:self.real_state_number]
        return indicator

    #
    def degree_distribution_sensitivity(self, matrix):
        indicator = np.empty(self.real_state_number, dtype=bool)
        for state_index in range(self.real_state_number):
            indicator[state_index] = False
            out_degree = matrix[state_index, :self.real_state_number]
            if np.sum(out_degree) > 0:
                sorted_degree = - np.sort(-out_degree)
                if sorted_degree[1] > 0:
                    if sorted_degree[0] / sorted_degree[1] < self.distribution_threshold:
                        indicator[state_index] = True
        return indicator
