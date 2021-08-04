import numpy as np
from tools.general_tools import GeneralTools
from tools.noise import Noise
import config.folder_and_file_names as config
from config.parameter_carrier import ParameterCarrier


class GuidePost:

    #
    def __init__(self, state_index, cc: ParameterCarrier):
        self.cc = cc

        self.this_state = state_index
        self.last_step = np.array([])
        self.next_step = np.array([])
        self.order2_trans_matrix = np.array([])
        self.all_state_number = -1
        self.start_state = -1
        self.end_state = -1

    #
    def give_last_step(self, last_steps):
        self.last_step = last_steps

    #
    def give_next_step(self, next_steps):
        self.next_step = next_steps

    #
    def guidepost_set_up(self, neighbors, all_state_number, start_state_number, end_state_number):
        self.give_last_step(neighbors)
        self.give_next_step(neighbors)
        self.start_state = start_state_number
        self.end_state = end_state_number
        self.all_state_number = all_state_number
        self.order2_trans_matrix = np.zeros((all_state_number, all_state_number), dtype=np.float16)

    #
    def guidepost_add(self, last_step, next_step, trajectory_length):
        if isinstance(last_step, str):
            if last_step == 'start':
                self.add_start(next_step, trajectory_length)
            else:
                raise TypeError('start point should be start')
        elif isinstance(next_step, str):
            if next_step == 'end':
                self.add_end(last_step, trajectory_length)
            else:
                raise TypeError('end point should be end')
        else:
            divider = int(trajectory_length)
            if divider < 1:
                divider = 1
            self.order2_trans_matrix[last_step, next_step] =\
                self.order2_trans_matrix[last_step, next_step] + 1 / divider

    #
    def choose_direction(self, last_step, step_number_now, return_probability=False):
        gt1 = GeneralTools()
        cc1 = self.cc
        if isinstance(last_step, str):
            if last_step == 'start':
                inner_last = self.start_state
            else:
                raise TypeError('start point should  be start')
        else:
            inner_last = last_step
        candidates = np.arange(self.order2_trans_matrix.shape[0])
        # candidates[-1] = self.order2_trans_matrix.shape[0] - 1
        probability = self.order2_trans_matrix[inner_last, :]
        # if not cc1.emphasis_on_trajectories_before:
        probability = probability.copy()
        average_step = 4
        steps_more_than_normal = step_number_now - average_step
        if steps_more_than_normal < 1:
            steps_more_than_normal = 1
        # end_multiplier = np.sqrt(steps_more_than_normal)

        # end_multiplier = gt1.steps_more_than_normal_to_end_multiplier(steps_more_than_normal)
        probability = probability.astype(float)
        # probability[-1] = probability[-1] * end_multiplier
        if return_probability:
            return probability
        # probability = probability / np.sum(probability)
        # inner_result = np.random.choice(candidates, p=probability)
        # gt1 = GeneralTools()
        inner_result = gt1.draw_by_probability_without_an_element(candidates, probability, -2)
        # if inner_result == self.end_state:
        #     result = 'end'
        # else:
        #     result = inner_result
        result = inner_result
        return result

    #
    def add_start(self, next_state, trajectory_length):
        if next_state == 'end':
            next_state = self.end_state
        self.order2_trans_matrix[self.start_state, next_state] = \
            self.order2_trans_matrix[self.start_state, next_state] + 1 / trajectory_length

    #
    def add_end(self, last_state, trajectory_length):
        if last_state == 'start':
            last_state = self.start_state
        self.order2_trans_matrix[last_state, self.end_state] = \
            self.order2_trans_matrix[last_state, self.end_state] + 1 / trajectory_length

    #
    def add_noise(self):
        cc1 = self.cc
        noise1 = Noise()
        epsilon = cc1.total_epsilon * cc1.epsilon_partition[2]
        noisy_matrix = noise1.add_laplace(self.order2_trans_matrix, epsilon, 1, if_regularize=False)
        noisy_matrix[:, self.start_state] = np.zeros(self.all_state_number)
        noisy_matrix[self.end_state, :] = np.zeros(self.all_state_number)
        # if cc1.level2_neighborhood_noise_filter:
        #     filter_matrix = np.zeros((noisy_matrix.shape[0] - 2, noisy_matrix.shape[0] - 2), dtype=int)
        #     filter_matrix[np.ix_(self.last_step, self.next_step)] = 1
        #     noisy_matrix[:-2, :-2] = noisy_matrix[:-2, :-2] * filter_matrix
        noisy_matrix = noise1.positive_regulation_for_markov_matrix(noisy_matrix)
        self.order2_trans_matrix = noisy_matrix.astype(np.int)

    #
    def give_total_ends_value(self):
        end_value = np.sum(self.order2_trans_matrix[:, -1])
        return end_value

    #
    def multiply_ends(self, multiplier):
        self.order2_trans_matrix[:, -1] = self.order2_trans_matrix[:, -1] * multiplier
