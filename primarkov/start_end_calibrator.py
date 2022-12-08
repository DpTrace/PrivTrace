import numpy as np
import math
import networkx as nx
from discretization.grid import Grid
import config.folder_and_file_names as config
# from tools.object_store import ObjectStore
from tools.general_tools import GeneralTools
from config.parameter_carrier import ParameterCarrier
import torch
import torch.optim as optim
import cvxpy as cp


class StartEndCalibrator:

    #
    def __init__(self, cc: ParameterCarrier):
        self.cc = cc
        self.non_zero_start_indices = np.array([], dtype=int)
        self.non_zero_start_values = np.array([])
        self.non_zero_end_indices = np.array([], dtype=int)
        self.non_zero_end_values = np.array([])
        self.all_usable_start_to_inner_indices_dict = np.array([], dtype=int)
        self.all_usable_end_to_inner_indices_dict = np.array([], dtype=int)
        self.real_indices_shortest_path_lengths = np.array([], dtype=int)
        self.inner_indices_shortest_path_lengths = np.array([], dtype=int)
        self.inner_indices_shortest_large_cell_paths_lengths = np.array([], dtype=int)
        self.inner_indices_arithmetic_mean_length = np.array([])
        self.distance_network = nx.Graph()
        self.state_number = -1
        self.total_trajectory_number = -1
        self.best_start_end_trip_distribution = np.array([])
        self.geo_lengths = np.array([])
        self.large_trans_indicator = np.array([])

    #
    def setup_network(self, grid: Grid):
        # step1 get central points of states
        central_point_gps = grid.usable_state_central_points()
        self.state_number = grid.usable_state_number
        self.total_trajectory_number = grid.trajectory_number
        for state_index in range(self.state_number):
            neighbors = grid.usable_state_neighbors(state_index)
            for neighbor_state in neighbors:
                distance = self.distance_of_central_points(central_point_gps, state_index, neighbor_state)
                self.distance_network.add_edge(state_index, neighbor_state, weight=distance)

    #
    def setup_direct_lengths(self, grid: Grid):
        self.geo_lengths = np.empty((self.non_zero_start_indices.size, self.non_zero_start_indices.size))
        start_state_number = self.non_zero_start_indices.size
        end_state_number = self.non_zero_end_indices.size
        central_point_gps = grid.usable_state_central_points()
        for inner_start_index in range(start_state_number):
            for inner_end_index in range(end_state_number):
                usable_start_index = self.non_zero_start_indices[inner_start_index]
                usable_end_index = self.non_zero_end_indices[inner_end_index]
                if usable_start_index == usable_end_index:
                    this_subcell_border = grid.level2_borders[usable_start_index, :]
                    self.geo_lengths[usable_start_index, usable_end_index] = np.sqrt(
                        (this_subcell_border[0] - this_subcell_border[1]) ** 2 + (
                                    this_subcell_border[3] - this_subcell_border[2]) ** 2) / 2
                else:
                    self.geo_lengths[usable_start_index, usable_end_index] = self.distance_of_central_points(
                        central_point_gps, usable_start_index, usable_end_index)

    #
    def distance_of_central_points(self, central_points_gps, state_index1, state_index2):
        state1_central_point_gps = central_points_gps[state_index1, :]
        state2_central_point_gps = central_points_gps[state_index2, :]
        displacement = state1_central_point_gps - state2_central_point_gps
        displacement_square = displacement ** 2
        distance = np.sqrt(np.sum(displacement_square))
        return distance

    #
    def setup_calibrator(self, grid: Grid, noisy_matrix, large_trans_indicator):
        gt1 = GeneralTools()
        self.setup_network(grid)
        self.large_trans_indicator = large_trans_indicator
        start_states_value = noisy_matrix[-2, :-2]
        end_states_value = noisy_matrix[:-2, -1]
        non_zero_start_values = start_states_value
        non_zero_end_values = end_states_value
        non_zero_start_indices = np.arange(start_states_value.size)
        non_zero_end_indices = np.arange(end_states_value.size)
        self.non_zero_start_indices = non_zero_start_indices
        self.non_zero_start_values = non_zero_start_values
        self.non_zero_end_indices = non_zero_end_indices
        self.non_zero_end_values = non_zero_end_values
        self.all_usable_start_to_inner_indices_dict = gt1.inverse_index_dict(self.state_number, non_zero_start_indices)
        self.all_usable_end_to_inner_indices_dict = gt1.inverse_index_dict(self.state_number, non_zero_end_indices)
        self.calculate_shortest_path_length(grid)

    #
    def calculate_shortest_path_length(self, grid: Grid):
        start_state_number = self.non_zero_start_indices.size
        end_state_number = self.non_zero_end_indices.size
        self.inner_indices_shortest_path_lengths = np.zeros((start_state_number, end_state_number)) - 1
        self.inner_indices_shortest_large_cell_paths_lengths = np.zeros((start_state_number, end_state_number)) - 1
        for inner_start_index in range(start_state_number):
            for inner_end_index in range(end_state_number):
                usable_start_index = self.non_zero_start_indices[inner_start_index]
                usable_end_index = self.non_zero_end_indices[inner_end_index]
                shortest_path = nx.dijkstra_path(self.distance_network, source=usable_start_index,
                                                 target=usable_end_index)
                large_cell_path = grid.non_repeat_large_cell_array_from_usable(shortest_path)
                self.inner_indices_shortest_path_lengths[inner_start_index, inner_end_index] = len(shortest_path)
                self.inner_indices_shortest_large_cell_paths_lengths[inner_start_index, inner_end_index] = len(
                    large_cell_path)
        self.inner_indices_arithmetic_mean_length = np.empty(self.inner_indices_shortest_path_lengths.shape)
        for i in range(self.inner_indices_shortest_path_lengths.shape[0]):
            for j in range(self.inner_indices_shortest_path_lengths.shape[1]):
                self.inner_indices_arithmetic_mean_length[i, j] = \
                    self.expect_length_in_geometric_length_distribution(self.inner_indices_shortest_path_lengths[i, j])

    #
    def break_constraints(self, start_end_trip_weights):
        cc1 = self.cc
        if isinstance(start_end_trip_weights, torch.Tensor):
            start_end_trip_weights = start_end_trip_weights.detach().numpy()
        # non zero constraint
        if (start_end_trip_weights <= 0).any():
            return True
        else:
            return False

    #
    def error_function(self, distribution):
        count_distribution = distribution
        start_error = self.start_distribution_error(count_distribution)
        end_error = self.end_distribution_error(count_distribution)
        total_error = start_error + end_error
        return total_error

    def start_distribution_error(self, distribution):
        if isinstance(distribution, torch.Tensor):
            error_sum = torch.zeros(1, requires_grad=True)
        elif isinstance(distribution, cp.Variable):
            error_sum = 0
        else:
            error_sum = np.zeros(1)
        for inner_start_index in range(self.non_zero_start_indices.size):
            error = self.error_of_inner_start_i(distribution, inner_start_index)
            error_sum = error_sum + error * error
        return error_sum

    def end_distribution_error(self, distribution):
        if isinstance(distribution, torch.Tensor):
            error_sum = torch.zeros(1, requires_grad=True)
        elif isinstance(distribution, cp.Variable):
            error_sum = 0
        else:
            error_sum = np.zeros(1)

        for inner_end_index in range(self.non_zero_end_indices.size):
            error = self.error_of_inner_end_i(distribution, inner_end_index)
            error_sum = error_sum + error * error
        return error_sum

    def expect_length_in_geometric_length_distribution(self, shortest_length):
        exp_of_two = np.exp(-(1 / shortest_length))
        re = 0.5 * shortest_length * (1 - exp_of_two) + np.exp(-((shortest_length + 1) / shortest_length)) * 1 / (
                    (1 - exp_of_two) ** 2)
        return re

    #
    def error_of_inner_start_i(self, distribution, inner_start_index):
        length_normalized_noisy_start_frequency = self.non_zero_start_values[inner_start_index]
        if isinstance(distribution, torch.Tensor):
            length_normalized_count = torch.zeros(1, requires_grad=True)
        elif isinstance(distribution, cp.Variable):
            length_normalized_count = 0
        else:
            length_normalized_count = np.zeros(1)
        for inner_end_index in range(self.non_zero_end_indices.size):
            length = self.inner_indices_shortest_path_lengths[inner_start_index, inner_end_index]
            length = self.expect_length_in_geometric_length_distribution(length)
            length_normalized_count = length_normalized_count + distribution[
                inner_start_index, inner_end_index] / length
        total_error = length_normalized_count - length_normalized_noisy_start_frequency
        return total_error

    def error_of_inner_end_i(self, distribution, inner_end_index):
        length_normalized_noisy_end_frequency = self.non_zero_end_values[inner_end_index]
        if isinstance(distribution, torch.Tensor):
            length_normalized_count = torch.zeros(1, requires_grad=True)
        elif isinstance(distribution, cp.Variable):
            length_normalized_count = 0
        else:
            length_normalized_count = np.zeros(1)
        for inner_start_index in range(self.non_zero_start_indices.size):
            length = self.inner_indices_shortest_path_lengths[inner_start_index, inner_end_index]
            length = self.expect_length_in_geometric_length_distribution(length)
            length_normalized_count = length_normalized_count + distribution[
                inner_start_index, inner_end_index] / length
        total_error = length_normalized_count - length_normalized_noisy_end_frequency
        return total_error

    #
    def attractiveness_of_states(self, noisy_matrix):
        useful_degrees = noisy_matrix[:-2, :-2]
        attractiveness = np.zeros(useful_degrees.shape[0])
        for i in range(useful_degrees.shape[0]):
            attractiveness_of_i = (np.sum(useful_degrees[i, :]) + np.sum(useful_degrees[:, i])) / 2
            attractiveness[i] = attractiveness_of_i
        return attractiveness

    def distribution_optimization_with_simple_gravity_model2(self, noisy_matrix, loose_parameter=20):
        discrete_lengths = self.inner_indices_shortest_path_lengths
        row_distribution = cp.Variable(self.non_zero_start_indices.size)
        objective = cp.Minimize(cp.square(cp.sum(row_distribution) - self.total_trajectory_number))
        attractiveness = self.attractiveness_of_states(noisy_matrix)
        distribution_weights = np.tile(attractiveness, (attractiveness.shape[0], 1)) / self.geo_lengths
        normalized_weights = distribution_weights / np.linalg.norm(distribution_weights, axis=1, ord=2)
        no_weights_distribution = row_distribution.T @ np.ones(attractiveness.shape[0])
        distribution = cp.multiply(normalized_weights, no_weights_distribution)
        expected_divided_start = cp.sum(cp.multiply(distribution, 1 / discrete_lengths), axis=0)
        expected_divided_end = cp.sum(cp.multiply(distribution, 1 / discrete_lengths), axis=1)
        constraints = [distribution >= 0,
                       expected_divided_start <= self.non_zero_start_values + loose_parameter,
                       expected_divided_start >= self.non_zero_start_values - loose_parameter,
                       expected_divided_end <= self.non_zero_end_values + loose_parameter,
                       expected_divided_end >= self.non_zero_end_values - loose_parameter]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        row_distribution_v = row_distribution.value
        if not isinstance(row_distribution_v, np.ndarray):
            return None
        else:
            all_distribution = np.matmul(np.ones((attractiveness.shape[0], 1)), row_distribution_v) * normalized_weights
            return all_distribution

    #
    def grades_of_discrete_lengths(self, discrete_lengths):
        min_length = np.min(discrete_lengths)
        max_length = np.max(discrete_lengths)
        level12_thre = min_length + (max_length - min_length) / 3
        level23_thre = min_length + 2 * (max_length - min_length) / 3
        level1_indicator = discrete_lengths < level12_thre
        level2_indicator = np.logical_and(discrete_lengths < level23_thre, discrete_lengths >= level12_thre)
        level3_indicator = discrete_lengths >= level23_thre
        graded_lengths = np.zeros(discrete_lengths.shape)
        graded_lengths[level1_indicator] = 1
        graded_lengths[level2_indicator] = 2
        graded_lengths[level3_indicator] = 3
        return graded_lengths

    #
    def distribution_optimization_with_simple_gravity_model(self, noisy_matrix, loose_parameter=20):
        discrete_lengths = self.inner_indices_shortest_path_lengths
        row_distribution = cp.Variable((1, self.non_zero_start_indices.size))
        attractiveness = self.attractiveness_of_states(noisy_matrix)
        distribution_weights = np.tile(attractiveness, (self.non_zero_start_indices.size, 1)) / np.sqrt(self.geo_lengths)
        normalized_weights = (distribution_weights.transpose() / np.linalg.norm(distribution_weights, axis=1, ord=1)).transpose()
        no_weights_distribution = np.ones((attractiveness.shape[0], 1)) @ row_distribution
        distribution = cp.multiply(normalized_weights, no_weights_distribution)
        divided_weights = cp.multiply(distribution, 1 / discrete_lengths)
        expected_divided_start = cp.sum(divided_weights, axis=1)
        expected_divided_end = cp.sum(divided_weights, axis=0)
        total_trajectory_number_error = cp.square(cp.sum(distribution) - self.total_trajectory_number)
        objective = cp.Minimize(cp.norm(expected_divided_start - self.non_zero_start_values, 2) + cp.norm(expected_divided_end - self.non_zero_end_values, 2))
        constraints = [distribution >= 0,
                       total_trajectory_number_error <= loose_parameter]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        distribution_v = distribution.value
        return distribution_v

    #
    def distribution_optimization_with_simple_gravity_model3(self, noisy_matrix, loose_parameter=20):
        discrete_lengths = self.inner_indices_shortest_path_lengths
        row_distribution = cp.Variable((1, self.non_zero_start_indices.size))
        attractiveness = self.attractiveness_of_states(noisy_matrix)
        distribution_weights = np.tile(attractiveness, (self.non_zero_start_indices.size, 1)) / np.sqrt(self.geo_lengths)
        normalized_weights = distribution_weights / np.linalg.norm(distribution_weights, axis=1, ord=2)
        no_weights_distribution = np.ones((attractiveness.shape[0], 1)) @ row_distribution
        distribution = cp.multiply(normalized_weights, no_weights_distribution)
        divided_weights = cp.multiply(distribution, 1 / discrete_lengths)
        expected_divided_start = cp.sum(divided_weights, axis=0)
        expected_divided_end = cp.sum(divided_weights, axis=1)
        total_trajectory_number_error = cp.square(cp.sum(distribution) - self.total_trajectory_number)
        objective = cp.Minimize(cp.norm(expected_divided_start - self.non_zero_start_values, 2) + cp.norm(expected_divided_end - self.non_zero_end_values, 2))
        constraints = [distribution >= 0,
                       total_trajectory_number_error <= loose_parameter]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        distribution_v = distribution.value
        return distribution_v

    def distribution_optimization_torch(self):
        distribution = Variable(torch.randn([self.non_zero_start_indices.size, self.non_zero_end_indices.size]))
        distribution = torch.abs(distribution)
        distribution = distribution / distribution.sum() * self.total_trajectory_number
        distribution.requires_grad = True
        while self.break_constraints(distribution):
            distribution = Variable(torch.randn([self.non_zero_start_indices.size, self.non_zero_end_indices.size]))
            distribution = torch.abs(distribution)
            distribution = distribution / distribution.sum() * self.total_trajectory_number
            distribution.requires_grad = True
        distribution.requires_grad = True
        optimizer = optim.Adam([distribution], lr=0.0001)
        num_epochs = 10000
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.error_function(distribution)
            loss.backward()
            optimizer.step()

            if self.break_constraints(distribution):
                break
            if epoch % 50 == 0:
                print(loss)
        return distribution.detach().numpy()

    #
    def distribution_optimization_cvxpy(self):
        distribution = cp.Variable((self.non_zero_start_indices.size, self.non_zero_end_indices.size))
        objective = cp.Minimize(self.error_function(distribution))
        constraints = [distribution >= 0,
                       np.sum(distribution) == self.total_trajectory_number]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return distribution

    #
    def distribution_optimization_cvxpy12(self, loose_parameter=20):
        distribution = cp.Variable((self.non_zero_start_indices.size, self.non_zero_end_indices.size))
        objective = cp.Minimize(cp.square(cp.sum(distribution) - self.total_trajectory_number))
        lengths = self.inner_indices_shortest_path_lengths
        lengths[lengths < 1] = 1
        constraints = [distribution >= 0,
                       cp.sum(cp.multiply(distribution, 1 / lengths),
                              axis=0) <= self.non_zero_start_values + loose_parameter,
                       cp.sum(cp.multiply(distribution, 1 / lengths),
                              axis=0) >= self.non_zero_start_values - loose_parameter,
                       cp.sum(cp.multiply(distribution, 1 / lengths),
                              axis=1) <= self.non_zero_end_values + loose_parameter,
                       cp.sum(cp.multiply(distribution, 1 / lengths),
                              axis=1) >= self.non_zero_end_values - loose_parameter]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        return distribution.value

    #
    def distribution_optimization_cvxpy2(self, loose_parameter=20):
        lengths = self.inner_indices_shortest_path_lengths
        lengths = lengths * self.large_trans_indicator
        lengths[lengths < 0.01] = 0.01
        distribution = cp.Variable((self.non_zero_start_indices.size, self.non_zero_end_indices.size))
        start_error = cp.norm(cp.sum(cp.multiply(distribution, 1 / lengths), axis=1) - self.non_zero_start_values)
        end_error = cp.norm(cp.sum(cp.multiply(distribution, 1 / lengths), axis=0) - self.non_zero_end_values)
        objective = cp.Minimize(start_error + end_error)
        constraints = [distribution >= 0,
                       cp.square(cp.sum(distribution) - self.total_trajectory_number) <= loose_parameter ** 2]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS)
        except:
            prob.solve(solver=cp.SCS)
        finally:
            if distribution.value is None:
                prob.solve(solver=cp.SCS)
            return distribution.value

    #
    def optimized_non_length_divided_distribution(self, divided_distribution):
        non_divided_distribution = divided_distribution
        non_divided_distribution = \
            non_divided_distribution / np.sum(non_divided_distribution) * self.total_trajectory_number
        non_divided_distribution[non_divided_distribution < 0.8] = 0
        non_divided_distribution = \
            non_divided_distribution / np.sum(non_divided_distribution) * self.total_trajectory_number
        return non_divided_distribution

    #
    def distribution_calibration(self, grid: Grid, noisy_matrix, large_trans_indicator):
        cc = self.cc
        self.setup_calibrator(grid, noisy_matrix, large_trans_indicator)
        divided_distribution = self.distribution_optimization_cvxpy2()
        iter_turns = 0
        while (divided_distribution is None) and (iter_turns < 10):
            divided_distribution = self.distribution_optimization_cvxpy2()
            iter_turns = iter_turns + 1
        loose_multiplier = 2
        while divided_distribution is None:
            divided_distribution = self.distribution_optimization_cvxpy2(loose_parameter=loose_multiplier ** 2)
            loose_multiplier = loose_multiplier + 1
        non_length_divided_distribution = self.optimized_non_length_divided_distribution(divided_distribution)
        return non_length_divided_distribution

    #
    def distribution_calibration_gravity_model_version(self, grid: Grid, noisy_matrix, large_trans_indicator):
        cc = self.cc
        self.setup_calibrator(grid, noisy_matrix, large_trans_indicator)
        divided_distribution = self.distribution_optimization_with_simple_gravity_model(noisy_matrix)
        iter_turns = 0
        while (divided_distribution is None) and (iter_turns < 10):
            divided_distribution = self.distribution_optimization_with_simple_gravity_model(noisy_matrix)
            iter_turns = iter_turns + 1
        loose_multiplier = 2
        while divided_distribution is None:
            divided_distribution = self.distribution_optimization_with_simple_gravity_model(noisy_matrix,
                                                                                            loose_parameter=loose_multiplier * 20)
            loose_multiplier = loose_multiplier + 1
        non_length_divided_distribution = self.optimized_non_length_divided_distribution(divided_distribution)
        return non_length_divided_distribution
