import numpy as np


class GeneralTools:

    #
    def __init__(self):
        pass

    # this function
    def get_bin(self, start, end, bin_number: int) -> np.ndarray:
        if type(bin_number) is not int:
            try:
                non_integer_part = bin_number % 1
                if non_integer_part != 0:
                    raise ValueError('wrong number of bins')
            except TypeError:
                raise TypeError('type of the bin number is wrong')
        interval = (end - start) / bin_number
        float_overflow_revise = 0.0000001
        bin1 = np.arange(start, end + interval - float_overflow_revise, interval)
        return bin1

    # this function digitize a point sequence
    # parameter outlier_handling means how to deal with data in array1 that is not in any bins i.e. in left or right of
    # the bin1 array. then if outlier_handling is 'label', the indices label outlier by -1 in indices returned
    # if outlier_handling is 'error' , the program reports an error.
    def get_bin_index(self, array1: np.ndarray, bin1: np.ndarray, outlier_handling: str = 'label') -> np.ndarray:
        if not isinstance(array1, np.ndarray):
            raise TypeError('type of given array is wrong')
        if not isinstance(bin1, np.ndarray):
            raise TypeError('type of given bin is wrong')
        indices = np.searchsorted(bin1, array1)
        max_bin_index = bin1.size - 1
        left_outlier = (indices <= 0)
        if left_outlier.any():
            left_outlier_indices = np.arange(indices.size)[left_outlier]
            for index in left_outlier_indices:
                continuous_value = array1[index]
                if bin1[0] - continuous_value < 0.0001:
                    indices[index] = 1
        right_outlier = (indices > max_bin_index)
        if right_outlier.any():
            right_outlier_indices = np.arange(indices.size)[right_outlier]
            for index in right_outlier_indices:
                continuous_value = array1[index]
                if continuous_value - bin1[-1] < 0.0001:
                    indices[index] = max_bin_index
        outlier_indicator = (indices <= 0) | (indices > max_bin_index)
        indices = indices - 1
        if outlier_handling == 'label':
            indices[outlier_indicator] = -1
        if outlier_handling == 'error':
            if outlier_indicator.any():
                raise ValueError('array to digitize has outlier, which is illegal')
        return indices

    # this function gives result of digitize of a 2-dimensional array by two bins. it is how a point is digitized.
    # be careful, the result is y and x, not x and y
    def get_points_bin_index(self, point_array: np.ndarray, x_bin: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
        x_result = self.get_bin_index(point_array[:, 0], x_bin)
        y_result = self.get_bin_index(point_array[:, 1], y_bin)
        column_result = np.reshape(x_result, (-1, 1))
        row_result = np.reshape(y_result, (-1, 1))
        combine_result = np.concatenate((row_result, column_result), axis=1)
        return combine_result

    # this function calculate frequency of full number cells given cell index, cell frequency and whole cell number.
    def whole_frequency(self, cell_indices: np.ndarray, cell_frequency_array: np.ndarray,
                        full_cell_number: int) -> np.ndarray:
        whole_frequency_array = np.zeros(full_cell_number)
        for index_for_cell_indices in range(cell_indices.size):
            cell_index = cell_indices[index_for_cell_indices]
            cell_frequency = cell_frequency_array[index_for_cell_indices]
            whole_frequency_array[cell_index] = cell_frequency
        return whole_frequency_array

    #
    def unreapted_int_array(self, sequence: np.ndarray):
        index_array = []
        frequency_array = []
        index_array.append(sequence[0])
        repeat_counter = 0
        for point1 in sequence:
            if point1 == index_array[-1]:
                repeat_counter += 1
            else:
                index_array.append(point1)
                frequency_array.append(repeat_counter)
                repeat_counter = 1
        frequency_array.append(repeat_counter)
        index_array = np.asarray(index_array, dtype=int)
        frequency_array = np.asarray(frequency_array, dtype=int)
        return index_array, frequency_array

    # this function calculates density giving a single index array
    def density_of_single_array(self, state_number: int, state_array: np.ndarray) -> np.ndarray:
        indices, frequency = np.unique(state_array, return_counts=True)
        wrong_cell_index = (indices >= state_number)
        if wrong_cell_index.any():
            raise IndexError('array has wrong cell index')
        whole_cell_frequency = self.whole_frequency(indices, frequency, state_number)
        return whole_cell_frequency

    # this function transfers a list of positions into a list of indices
    def transfer_set_of_elements(self, input_list, reference_dict: dict):
        if isinstance(input_list, list):
            output_list = []
            for input_element in input_list:
                if isinstance(input_element, np.ndarray):
                    input_element = tuple(input_element)
                index = reference_dict[input_element]
                index.append(index)
        if isinstance(input_list, dict):
            output_list = {}
            for key in input_list:
                position = input_list[key]
                if position is not False:
                    position = tuple(position)
                    output_value = reference_dict[position]
                    output_list[key] = output_value
        else:
            raise TypeError('wrong data to transfer')
        return output_list

    # this function
    def draw_by_probability(self, candidates, probability):
        if candidates.size == 0:
            raise ValueError('candidates can not be empty')
        if np.min(probability) < 0:
            raise ValueError('probability should not be nagative')
        pro_sum = np.sum(probability)
        if np.isinf(pro_sum):
            print('this is inf')
            print(probability)
            print(np.arange(probability.size)[probability > 1000000000])
        if pro_sum == 0:
            probability = probability + 1 / probability.size
        probability = probability / np.sum(probability)
        if np.isnan(probability).any():
            result = candidates[-1]
        else:
            try:
                result = np.random.choice(candidates, p=probability)
            except ValueError:
                print('probabilities do not sum to 1, sum is {}'.format(np.sum(probability)))
                print(pro_sum)
                print(probability)
        return result

    #
    def draw_by_probability_without_an_element(self, candidates, probability, element_index_to_delete):
        if candidates.size == 0:
            raise ValueError('candidates can not be empty')
        if np.min(probability) < 0:
            raise ValueError('probability should not be nagative')
        element_indicator = np.ones(candidates.size, dtype=bool)
        if isinstance(element_index_to_delete, int):
            element_indicator[element_index_to_delete] = False
        else:
            for index_to_del in element_index_to_delete:
                element_indicator[index_to_del] = False
        good_candicates = candidates[element_indicator]
        good_probability = probability[element_indicator]
        result = self.draw_by_probability(good_candicates, good_probability)
        return result

    #
    def random_pick_element(self, element_array: np.ndarray):
        probability = np.zeros(element_array.shape) + 1 / element_array.size
        picked_element = np.random.choice(element_array, p=probability)
        return picked_element

    #
    def sample_from_interval(self, low, high):
        length = high - low
        random_number = np.random.random_sample()
        result = low + random_number * length
        return result

    #
    def matrix_relative_error_with_order_1(self, real_matrix, matrix_to_compare):
        real_matrix = real_matrix.reshape(-1)
        real_matrix = real_matrix / np.sum(real_matrix)
        matrix_to_compare = matrix_to_compare.reshape(-1)
        matrix_to_compare = matrix_to_compare / np.sum(matrix_to_compare)
        total_amount = np.sum(real_matrix)
        error_matrix = matrix_to_compare - real_matrix
        error = np.linalg.norm(error_matrix, ord=1)
        error_ratio = error / total_amount
        return error_ratio

    #
    def one_dimensional_bin_density(self, array: np.ndarray, bins: np.ndarray):
        density = np.zeros(bins.size - 1, dtype=int)
        bin_index_of_array_point = self.get_bin_index(array, bins)
        existing_bins, bin_frequency = np.unique(bin_index_of_array_point, return_counts=True)
        for index_of_element_in_existing_bins in range(existing_bins.size):
            this_bin = existing_bins[index_of_element_in_existing_bins]
            if this_bin >= 0:
                this_frequency = bin_frequency[index_of_element_in_existing_bins]
                density[this_bin] = this_frequency
        return density

    def steps_more_than_normal_to_end_multiplier(self, steps_more_than_normal):
        end_multiplier = 1
        return end_multiplier

    def non_zero_values(self, array):
        all_indices = np.arange(array.size)
        indicator = (array > 0)
        non_zero_values = array[indicator]
        indices = all_indices[indicator]
        return non_zero_values, indices

    # this function gives usable to real index dict
    def inverse_index_dict(self, all_number: int, original_dict: np.ndarray):
        inverse_dict = np.zeros(all_number, dtype=int)
        for original_index in range(original_dict.size):
            inverse_index = original_dict[original_index]
            if inverse_index >= 0:
                inverse_dict[inverse_index] = original_index
        return inverse_dict

    #
    def neighbors_usable_indices_of_states(self, states, neighbor_dict):
        neighbor_pool = []
        for state in states:
            neighbor_pool = neighbor_pool + neighbor_dict[state].tolist()
        neighbor_pool = np.array(neighbor_pool, dtype=int)
        neighbor_pool = np.unique(neighbor_pool)
        return neighbor_pool

    #
    def level1_array_length(self, array, grid):
        subcell_array = grid.usable_subcell_index_to_real_index_dict[array]
        level1_array = grid.level2_subcell_to_large_cell_dict[subcell_array]
        gt1 = GeneralTools()
        unrepeated_array = gt1.unreapted_int_array(level1_array)[0]
        length = unrepeated_array.size
        return length

    #
    def full_bridge_between_position(self, start_position, end_position):
        if np.linalg.norm((start_position - end_position), ord=1) <= 1:
            return False
        else:
            bridge_length = int(np.linalg.norm((start_position - end_position), ord=1) - 1)
            bridge = np.empty((bridge_length, 2), dtype=int)
            this_step_position = start_position
            for i in range(bridge_length):
                one_step_of_bridge = self.one_step_bridge_between_position(this_step_position, end_position)
                bridge[i, :] = one_step_of_bridge
                this_step_position = one_step_of_bridge
        return bridge

    #
    def one_step_bridge_between_position(self, start_position, end_position):
        probability = 0.8
        if np.linalg.norm((start_position - end_position), ord=1) <= 1:
            raise Exception('can not bridge between two neighbor state')
        else:
            x_displacement = end_position[0] - start_position[0]
            y_displacement = end_position[1] - start_position[1]
            if np.abs(x_displacement) > np.abs(y_displacement):
                direction = np.random.choice(np.array(['x', 'y']), p=[probability, 1-probability])
            else:
                direction = np.random.choice(np.array(['x', 'y']), p=[1 - probability, probability])
            if direction == 'x':
                next_step_index = self.one_step_in_a_dimension(start_position[0], end_position[0])
                return np.array([next_step_index, start_position[1]], dtype=int)
            else:
                next_step_index = self.one_step_in_a_dimension(start_position[1], end_position[1])
                return np.array([start_position[0], next_step_index], dtype=int)

    #
    def one_step_in_a_dimension(self, start_index, end_index):
        if start_index > end_index:
            return start_index - 1
        else:
            return start_index + 1

    #
    def check_arrays_shape(self, list_of_arrays: list, dimensions_to_check: np.ndarray):
        for dimension in dimensions_to_check:
            shapes = []
            for array in list_of_arrays:
                shapes.append(array.shape[dimension])
            unique_shape = np.unique(np.array(shapes))
            if unique_shape.size > 1:
                return False
        return True

    #
    def bonding_arrays(self, list_of_arrays: list):
        if not self.check_arrays_shape(list_of_arrays, np.array([1], dtype=int)):
            raise ValueError('Arrays must have same shape in axis 1')
        total_length = 0
        for array in list_of_arrays:
            total_length = total_length + array.shape[0]
        whole_array = np.empty((total_length, 2))
        start_position = 0
        cut_position = []
        for array in list_of_arrays:
            this_length = array.shape[0]
            end_position = start_position + this_length
            whole_array[start_position:end_position, :] = array
            cut_position.append(start_position)
            start_position = end_position
        cut_position.append(total_length)
        cut_position = np.array(cut_position, dtype=int)
        return whole_array, cut_position

    #
    def decompose_bonded_arrays_by_continuous_cut(self, bonded_arrays: np.ndarray, cut_position: np.ndarray):
        array_list = []
        cut_number = cut_position.size
        for cut_position_index in range(cut_number - 1):
            start_position = cut_position[cut_position_index]
            end_position = cut_position[cut_position_index + 1]
            array_list.append(bonded_arrays[start_position: end_position, :])
        return array_list

    #
    def decompose_bonded_arrays_by_discrete_cuts(self, bonded_arrays: np.ndarray, cut_position: list):
        array_list = []
        cut_number = len(cut_position)
        for cut_index in range(cut_number):
            cut = cut_position[cut_index]
            cut_start = cut[0]
            cut_end = cut[1]
            array_list.append(bonded_arrays[cut_start: cut_end, :])
        return array_list

    #
    def cut_by_points_to_segments(self, cut_position):
        discrete_cut_list = []
        for continuous_cut_index in range(cut_position.size - 1):
            start_continuous_position = cut_position[continuous_cut_index]
            end_continuous_position = cut_position[continuous_cut_index + 1]
            if start_continuous_position == end_continuous_position - 1:
                raise ValueError('no segment in some array')
            new_cut = np.array([start_continuous_position, end_continuous_position - 1], dtype=int)
            discrete_cut_list.append(new_cut)
        return discrete_cut_list

    #
    def border_of_trajectory_list(self, trajectory_list):
        south1 = 100000000
        north1 = -100000000
        west1 = 100000000
        east1 = -100000000
        trajectory_number = len(trajectory_list)
        for trajectory_index in range(trajectory_number):
            trajectory1 = trajectory_list[trajectory_index]
            arr = trajectory1
            if west1 > np.amin(arr[:, 0]):
                west1 = np.amin(arr[:, 0])
            if east1 < np.amax(arr[:, 0]):
                east1 = np.max(arr[:, 0])
            if south1 > np.amin(arr[:, 1]):
                south1 = np.amin(arr[:, 1])
            if north1 < np.amax(arr[:, 1]):
                north1 = np.amax(arr[:, 1])
        return np.array([north1, south1, west1, east1])

    #
    def intervals_overlap(self, interval1, interval2):
        if np.min(interval1) >= np.max(interval2):
            return False
        elif np.max(interval1) <= np.min(interval2):
            return False
        else:
            return True

    #
    def rec_overlap(self, rec1_border, rec2_border):
        rec1_y_interval = np.array([rec1_border[1], rec1_border[0]])
        rec1_x_interval = np.array([rec1_border[2], rec1_border[3]])
        rec2_y_interval = np.array([rec2_border[1], rec2_border[0]])
        rec2_x_interval = np.array([rec2_border[2], rec2_border[3]])
        return self.intervals_overlap(rec1_x_interval, rec2_x_interval) and self.intervals_overlap(rec1_y_interval, rec2_y_interval)








