import numpy as np
import config.folder_and_file_names as config
from config.parameter_carrier import ParameterCarrier
from data_preparation.trajectory import Trajectory
from data_preparation.trajectory_set import TrajectorySet
from discretization.grid import Grid
from tools.noise import Noise
from primarkov.sensitive_filter import Filter
from primarkov.guidepost import GuidePost
from primarkov.start_end_calibrator import StartEndCalibrator
import datetime
import copy


class MarkovModel:

    def __init__(self, cc: ParameterCarrier):
        self.cc = cc

        self.real_markov_matrix = np.array([])
        self.noisy_markov_matrix = np.array([])
        self.optimized_start_end_distribution = np.array([])
        self.shortest_lengths = np.array([])
        self.large_cell_lengths = np.array([])
        self.length_inside_large_cell = np.array([])
        self.start_state_index = -1
        self.end_state_index = -1
        self.subcell_number = -1
        self.all_state_number = -1
        self.grid = Grid(self.cc)
        self.calibrator = StartEndCalibrator(self.cc)

        self.guidepost_indicator = np.array([])
        self.guidepost_indices = np.array([])
        self.index_dict = np.array([])
        self.guidepost_set = []
        self.neighboring_matrix = np.array([])
        self.level1_length_threshold = np.array([])
        self.whole_length_thresholds = []
        self.large_trans_indicator = np.array([])

        #

    #
    def set_up_for_model(self, grid: Grid) -> None:
        subcell_number = grid.usable_state_number
        self.subcell_number = subcell_number
        self.start_state_index = subcell_number
        self.end_state_index = subcell_number + 1
        self.all_state_number = subcell_number + 2
        self.grid = grid

    # this function calculate markov function for a single trajectory
    def trajectory_markov_probability(self, trajectory1: Trajectory) -> np.ndarray:
        state_number = self.all_state_number
        start_state = self.start_state_index
        end_state = self.end_state_index
        trajectory_array = trajectory1.usable_simple_sequence
        markov_matrix = np.zeros((state_number, state_number))
        trajectory_length = trajectory_array.size
        for markov_transform_start in range(trajectory_length - 1):
            this_step_start_state = trajectory_array[markov_transform_start]
            this_step_end_state = trajectory_array[markov_transform_start + 1]
            markov_matrix[this_step_start_state, this_step_end_state] += 1
        transition_number_of_trajectory = trajectory_length + 1
        if transition_number_of_trajectory < 1:
            transition_number_of_trajectory = 1

        real_start_state = trajectory_array[0]
        real_end_state = trajectory_array[-1]
        markov_matrix[start_state, real_start_state] = 1
        markov_matrix[real_end_state, end_state] = 1
        
        markov_matrix = markov_matrix / transition_number_of_trajectory

        return markov_matrix

    # this function calculate markov transformation probability, usually first order.
    def calculate_markov_probability(self, trajectory_set: TrajectorySet) -> None:
        state_number1 = self.all_state_number
        markov_matrix = np.zeros((state_number1, state_number1))
        trajectory_list = trajectory_set.trajectory_list
        print('begin calculating matrix')
        print(datetime.datetime.now())
        for trajectory1 in trajectory_list:
            not_out_of_usable = not trajectory1.has_not_usable_index
            if not_out_of_usable:
                markov_matrix1 = self.trajectory_markov_probability(trajectory1)
                markov_matrix += markov_matrix1
        print('calculating ends')
        print(datetime.datetime.now())
        self.real_markov_matrix = markov_matrix

    # this function add noise to real markov matrix
    def noisy_markov(self):
        noise1 = Noise()
        cc1 = self.cc
        real_markov = self.real_markov_matrix
        total_epsilon = cc1.total_epsilon
        epsilon_partition_for_markov = cc1.epsilon_partition[1]
        epsilon_for_markov = total_epsilon * epsilon_partition_for_markov
        sensitivity = 1
        noisy_markov = noise1.add_laplace(real_markov, epsilon_for_markov, sensitivity, if_regularize=False)
        noisy_markov[:, self.start_state_index] = np.zeros(self.all_state_number)
        noisy_markov[self.end_state_index, :] = np.zeros(self.all_state_number)
        noisy_markov[self.start_state_index, self.end_state_index] = 0
        noisy_markov = noise1.positive_regulation_for_markov_matrix(noisy_markov, 'queue_minus')
        self.noisy_markov_matrix = noisy_markov

    #
    def get_filtered_sensitive_states(self):
        filter1 = Filter(self.cc)
        indicator = filter1.find_sensitive_state(self.noisy_markov_matrix)
        states = np.arange(indicator.size)[indicator]
        return states

    #
    def get_sensitive_state(self):
        filter1 = Filter(self.cc)
        indicator = filter1.find_sensitive_state(self.noisy_markov_matrix)
        indicator = np.concatenate((indicator, np.array([False, False])))
        indicator[self.start_state_index] = False
        indicator[self.end_state_index] = False
        self.guidepost_indicator = indicator
        self.sensitive_indices()

    #
    def sensitive_indices(self):
        indicator = self.guidepost_indicator
        indices = np.arange(indicator.size)[indicator]
        self.guidepost_indices = indices

    #
    def set_up_guideposts(self, grid):
        index_dict = np.zeros(self.all_state_number, dtype=int) - 1
        sensitive_indices = self.guidepost_indices
        gp_counter = 0
        for index in sensitive_indices:
            if_sensitive = self.guidepost_indicator[index]
            if if_sensitive:
                guidepost1 = GuidePost(index, self.cc)
                neighbors = grid.subcell_neighbors_usable_index[index]
                guidepost1.guidepost_set_up(neighbors, self.all_state_number, self.start_state_index, self.end_state_index)
                self.guidepost_set.append(guidepost1)
                index_dict[index] = gp_counter
                gp_counter = gp_counter + 1
        self.index_dict = index_dict

    #
    def give_guidepost_order2_info(self, trajectory_set1: TrajectorySet):
        for trajectory1 in trajectory_set1.trajectory_list:
            sequence = trajectory1.usable_simple_sequence
            trajectory_length = sequence.size
            for index_in_sequence in range(sequence.size):
                state_now = sequence[index_in_sequence]
                if index_in_sequence == 0:
                    state_previous = 'start'
                else:
                    previous_step_index = index_in_sequence - 1
                    state_previous = sequence[previous_step_index]
                if index_in_sequence == sequence.size - 1:
                    state_next = 'end'
                else:
                    next_step_index = index_in_sequence + 1
                    state_next = sequence[next_step_index]
                if_sensitive = self.guidepost_indicator[state_now]
                if if_sensitive:
                    guidepost_index = self.index_dict[state_now]
                    guidepost1 = self.guidepost_set[guidepost_index]
                    guidepost1.guidepost_add(state_previous, state_next, trajectory_length)

    #
    def give_neighboring_matrix(self, grid:Grid):
        subcell_number = self.subcell_number
        neighbors = grid.subcell_neighbors_usable_index
        matrix = np.empty((subcell_number, subcell_number), dtype=bool)
        for subcell_index1 in range(subcell_number):
            for subcell_index2 in range(subcell_number):
                matrix[subcell_index1, subcell_index2] = False
        for subcell_index in range(subcell_number):
            neeighbors_of_this_subcell = neighbors[subcell_index]
            for neighbor in neeighbors_of_this_subcell:
                matrix[subcell_index, neighbor] = True
        self.neighboring_matrix = matrix

    #
    def get_noisy_tran_pro_of_step_i(self, step_i):
        pro = self.noisy_markov_matrix[step_i, :]
        return pro.copy()

    #
    def add_noise_to_guidepost(self):
        for guidepost in self.guidepost_set:
            guidepost.add_noise()

    #
    def order1_and_2_end_consistency(self):
        for gp in self.guidepost_set:
            index_of_gp = gp.this_state
            order1_end_value = self.noisy_markov_matrix[index_of_gp, -1]
            order2_end_value = gp.give_total_ends_value()
            gp.multiply_ends(order1_end_value / order2_end_value * 1.5)

    #
    def start_end_trip_distribution_calibration(self):
        sec1 = StartEndCalibrator(self.cc)
        self.shortest_lengths = np.zeros((self.grid.usable_state_number, self.grid.usable_state_number))
        self.large_cell_lengths = np.zeros((self.grid.usable_state_number, self.grid.usable_state_number))
        self.large_trans_with_neighbors()
        optimized_distribution = sec1.distribution_calibration(self.grid, self.noisy_markov_matrix, self.large_trans_indicator)
        inner_start_index_to_usable = sec1.non_zero_start_indices
        inner_end_index_to_usable = sec1.non_zero_end_indices
        optimized_start_distribution = np.sum(optimized_distribution, axis=1)
        optimized_end_distribution = np.sum(optimized_distribution, axis=0)
        self.noisy_markov_matrix[-2, inner_start_index_to_usable] = optimized_start_distribution
        self.noisy_markov_matrix[inner_end_index_to_usable, -1] = self.noisy_markov_matrix[inner_end_index_to_usable, -1] * 1.3
        self.optimized_start_end_distribution = np.zeros(
            (self.grid.usable_state_number, self.grid.usable_state_number))
        for inner_row_index in range(inner_start_index_to_usable.size):
            for inner_column_index in range(inner_end_index_to_usable.size):
                row_index = inner_start_index_to_usable[inner_row_index]
                column_index = inner_end_index_to_usable[inner_column_index]
                distribution_value = optimized_distribution[inner_row_index, inner_column_index]
                shortest_length = sec1.inner_indices_shortest_path_lengths[inner_row_index, inner_column_index]
                large_cell_length = sec1.inner_indices_shortest_large_cell_paths_lengths[
                    inner_row_index, inner_column_index]
                self.optimized_start_end_distribution[row_index, column_index] = distribution_value
                self.shortest_lengths[row_index, column_index] = shortest_length
                self.large_cell_lengths[row_index, column_index] = large_cell_length
        self.length_inside_large_cell = self.shortest_lengths - self.large_cell_lengths
        self.calibrator = sec1

    #
    def whole_trajectory_len_threshold_and_weight(self, start_state):
        trip_weight = self.optimized_start_end_distribution[start_state, :]
        lengths = self.length_inside_large_cell[start_state, :]
        lengths = lengths[trip_weight > 0]
        trip_weight = trip_weight[trip_weight > 0]
        if np.sum(lengths) <= 0:
            return False
        len_thre = np.unique(lengths)
        len_thre = len_thre[len_thre > 0]
        keep_weight = np.zeros(len_thre.size)
        for i in range(len_thre.size):
            len1 = len_thre[i]
            weight = np.sum(trip_weight[lengths >= len1])
            keep_weight[i] = weight
        if keep_weight[0] > 0:
            keep_weight = keep_weight ** 2
            keep_weight = keep_weight / keep_weight[0]
            zero_weight_indicator = (keep_weight <= 0)
            if zero_weight_indicator.any():
                zero_indices = np.arange(zero_weight_indicator.size)[zero_weight_indicator]
                keep_weight[zero_weight_indicator] = keep_weight[zero_weight_indicator] +\
                                                 0.5 * keep_weight[(np.min(zero_indices) - 1)]
            return [len_thre, keep_weight]
        else:
            return False

    #
    def give_level1_length_thresholds(self):
        all_usable_state_number = self.grid.usable_state_number
        self.level1_length_threshold = np.empty(all_usable_state_number)
        for state_i in range(all_usable_state_number):
            # if self.cc.optimization_on:
            distribution_weights = self.optimized_start_end_distribution[state_i, :]
            # else:
            #     distribution_weights = self.give_weights_without_optimization(state_i)
            if np.sum(distribution_weights) <= 0:
                self.level1_length_threshold[state_i] = -1
            else:
                distribution = distribution_weights
                distribution = distribution / np.sum(distribution)
                average_len = 0
                for end_state_j in range(all_usable_state_number):
                    large_len = self.level1_length_of_two_usable_state(state_i, end_state_j) + np.sqrt(self.grid.level1_cell_number)
                    average_len = average_len + large_len * distribution[end_state_j]
                self.level1_length_threshold[state_i] = average_len

    #
    def give_weights_without_optimization(self, start_index):
        weights = np.zeros(self.grid.usable_state_number)
        for end_index in range(self.grid.usable_state_number):
            rough_len = self.level1_length_of_two_usable_state(start_index, end_index)
            weights[end_index] = 1 / rough_len
        return weights

    #
    def level1_length_of_two_usable_state(self, start_i, end_j):
        grid = self.grid
        usable_to_subcell = grid.usable_subcell_index_to_real_index_dict
        subcell_to_large_cell = grid.level2_subcell_to_large_cell_dict
        subcell_start = usable_to_subcell[start_i]
        subcell_end = usable_to_subcell[end_j]
        large_start = int(subcell_to_large_cell[subcell_start])
        large_end = int(subcell_to_large_cell[subcell_end])
        position_start = grid.level1_cell_position[large_start, :]
        position_end = grid.level1_cell_position[large_end, :]
        distance = position_end - position_start
        length = np.abs(distance[0]) + np.abs(distance[1]) + 1
        return length

    #
    def give_whole_length_thresholds(self):
        all_usable_state_number = self.grid.usable_state_number
        for i in range(all_usable_state_number):
            if np.sum(self.optimized_start_end_distribution[i, :]) <= 0:
                self.whole_length_thresholds.append(False)
            else:
                threshold = self.whole_trajectory_len_threshold_and_weight(i)
                self.whole_length_thresholds.append(threshold)

    #
    def large_trans_with_neighbors(self):
        original_large_trans = self.real_markov_matrix[:-2, :-2] > 0
        self.large_trans_indicator = self.find_two_step_distribution_neighbors(original_large_trans)

    #
    def find_one_step_distribution_neighbors(self, original_distribution):
        grid = self.grid
        result = grid.add_neighbors_to_distribution(original_distribution)
        return result

    #
    def find_two_step_distribution_neighbors(self, original_distribution):
        grid = self.grid
        result = grid.add_neighbors_to_distribution(original_distribution)
        result = grid.add_neighbors_to_distribution(result)
        return result

    def model_building(self, trajectory_set1: TrajectorySet, grid: Grid) -> None:
        self.set_up_for_model(grid)
        self.give_neighboring_matrix(grid)
        self.calculate_markov_probability(trajectory_set1)
        self.noisy_markov()
        pass

    #
    def model_filtering(self, trajectory_set1: TrajectorySet, grid: Grid):
        self.start_end_trip_distribution_calibration()
        self.give_level1_length_thresholds()
        self.get_sensitive_state()
        self.set_up_guideposts(grid)
        self.give_guidepost_order2_info(trajectory_set1)
        self.add_noise_to_guidepost()
        self.order1_and_2_end_consistency()
        pass

