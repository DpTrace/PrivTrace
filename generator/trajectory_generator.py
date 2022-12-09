import numpy as np
from tools.general_tools import GeneralTools
from primarkov.mar_model import MarkovModel
from config.parameter_carrier import ParameterCarrier
import datetime


class Generator:

    #
    def __init__(self, cc: ParameterCarrier):
        self.cc = cc
        # this parameter decides if all states are lingering in trajectory. If unique states number / trajectory length
        # smaller than this threshold, all states in trajectory are circling. So abundant this trajectory.
        self.lingering_if_all_circle_threshold = 0.2
        # this parameter decides if the remaining trajectories are lingering. If states of former ratio has frequency
        # summing to total frequency times this ratio, then this trajectory is lingering.
        self.lingering_weight_threshold = 0.6

        self.markov_model = MarkovModel(self.cc)
        self.average_length = -1
        self.level1_length_threshold_value = -1
        self.average_subdividing_number = -1
        self.simple_level2_length_threshold_value = -1
        self.total_in_degree = np.array([])
        self.total_out_degree = np.array([])
        self.latest_road_to_end = []
        self.strong_end = np.array([])
        self.strong_end_neighbors = np.array([])
        pass

    #
    def load_generator(self, mar_model):
        self.markov_model = mar_model
        self.average_length = np.average(
            self.markov_model.grid.level2_subdividing_parameter) + self.markov_model.grid.level1_cell_number / 2
        subdividing_para = np.sort(self.markov_model.grid.level2_subdividing_parameter)
        # i = np.int(np.floor(subdividing_para.size / 3))
        self.average_subdividing_number = np.average(subdividing_para[-5:])
        self.level1_length_threshold()
        self.simple_whole_trajectory_len_threshold(self.level1_length_threshold_value)
        self.total_in_degree = np.sum(self.markov_model.noisy_markov_matrix[0:-2, 0:-2], axis=0)
        self.total_out_degree = np.sum(self.markov_model.noisy_markov_matrix[0:-2, 0:-2], axis=1)
        self.find_neighbors_for_strong_end()

    #
    def find_neighbors_for_strong_end(self):
        cc1 = self.cc
        grid = self.markov_model.grid
        end_weights = self.markov_model.noisy_markov_matrix[:-2, -1]
        big_shot_indicator = (end_weights > 0.1 * np.sum(end_weights))
        strong_end = np.arange(big_shot_indicator.size)[big_shot_indicator]
        self.strong_end = strong_end
        all_neighbors = []
        for one_end_index in self.strong_end:
            neighbors = self.get_multilayer_neighbors(one_end_index)
            all_neighbors = all_neighbors + neighbors[0].tolist()
        self.strong_end_neighbors = np.unique(np.array(all_neighbors))

    #
    def generate_no_gp_step(self, this_step, step_now, return_probability=False):
        cc1 = self.cc
        gt1 = GeneralTools()
        mar_m = self.markov_model
        all_state = mar_m.all_state_number
        states = np.arange(all_state)
        probability = mar_m.get_noisy_tran_pro_of_step_i(this_step)
        if return_probability:
            return probability
        if step_now == 0:
            result = gt1.draw_by_probability_without_an_element(states, probability, [-2, -1])
        else:
            result = gt1.draw_by_probability_without_an_element(states, probability, -2)
        return result

    #
    def check_if_neighbor(self, this_step, step_to_check):
        neighbor_matrix = self.markov_model.neighboring_matrix
        if_neighbor = neighbor_matrix[this_step, step_to_check]
        return if_neighbor

    #
    def generate_one_step(self, this_step, last_step, step_number_now, neighbor_check=False, return_probability=False):
        mar_m = self.markov_model
        guide_post_indicator = mar_m.guidepost_indicator
        guidepost_used_actually = True
        use_guidepost = False

        if guide_post_indicator[this_step]:
            use_guidepost = True
        if use_guidepost:
            guide_post_index = mar_m.index_dict[this_step]
            gp1 = mar_m.guidepost_set[guide_post_index]
            next_step = gp1.choose_direction(last_step, step_number_now, return_probability=return_probability)
            if np.sum(next_step) == 0:
                next_step = self.generate_no_gp_step(this_step, step_number_now, return_probability=return_probability)
        else:
            next_step = self.generate_no_gp_step(this_step, step_number_now, return_probability=return_probability)
            guidepost_used_actually = False
        return next_step, guidepost_used_actually

    #
    def generate_no_guidepost_one_step(self, this_step, step_number_now, neighbor_check=False,
                                       return_probability=False):
        next_step = self.generate_no_gp_step(this_step, step_number_now, return_probability=return_probability)
        if return_probability:
            return next_step
        if neighbor_check:
            if next_step == self.markov_model.end_state_index:
                neighbor_check = True
            else:
                neighbor_check = self.check_if_neighbor(this_step, next_step)
            if neighbor_check:
                return next_step
            else:
                return False
        else:
            return next_step

    #
    def get_multilayer_neighbors(self, end_state):
        grid = self.markov_model.grid
        gt1 = GeneralTools()
        cc1 = self.cc
        multilayer_neighbors = []
        # neighbor_multiplier = cc1.neighbor_multiplier
        to_know_my_neighbor = np.array([end_state], dtype=int)
        for i in range(3):
            neighbors = gt1.neighbors_usable_indices_of_states(to_know_my_neighbor, grid.subcell_neighbors_usable_index)
            multilayer_neighbors.append(neighbors)
            to_know_my_neighbor = neighbors
        return multilayer_neighbors

    #
    def level1_length_threshold(self):
        cc1 = self.cc
        grid = self.markov_model.grid
        level1_state_number = grid.level1_cell_number
        level1_para = np.sqrt(level1_state_number)
        threshold = np.int(np.floor(level1_para * 0.8))
        if threshold < 0:
            threshold = 0
        self.level1_length_threshold_value = threshold

    def simple_whole_trajectory_len_threshold(self, level1_len_threshold):
        # grid = self.markov_model.grid
        average_subdividing_number = self.average_subdividing_number
        # whole_threshold = np.int(np.ceil(average_subdividing_number)) * 1.5
        whole_threshold = np.int(np.ceil(average_subdividing_number))
        if whole_threshold < level1_len_threshold:
            whole_threshold = level1_len_threshold
        self.simple_level2_length_threshold_value = whole_threshold

    #
    def keep_this_trajectory_with_level1_threshold(self, trajectory, level1_len_threshold, filtered_time):
        cc1 = self.cc
        gt1 = GeneralTools()
        grid = self.markov_model.grid
        level1_divide_number = np.sqrt(grid.level1_cell_number)
        level1_step_number = gt1.level1_array_length(trajectory, grid)
        if level1_len_threshold is False:
            if level1_len_threshold > level1_divide_number:
                return False
            else:
                return True
        different_large_cell_number = np.unique(grid.level2_subcell_to_large_cell_dict[trajectory]).size
        try_to_drop = False
        if different_large_cell_number / level1_step_number > 0.6:
            if level1_step_number > level1_len_threshold:
                try_to_drop = True
        else:
            if level1_step_number > 2 * level1_len_threshold:
                try_to_drop = True
        if try_to_drop:
            pass_probability = filtered_time / level1_divide_number
            if pass_probability > 0.95:
                pass_probability = 0.95
            drop_probability = 1 - pass_probability
            drop = np.random.choice(np.array([True, False]), p=np.array([drop_probability, 1 - drop_probability]))
            if drop:
                return False
            # else:
            #     return True
        else:
            return True

    def get_level1_threshold_in_use(self, start_index):
        thr = self.markov_model.level1_length_threshold[start_index]
        if thr <= 0:
            thr = self.level1_length_threshold_value
        return thr

    def generate_trajectory(self, neighbor_check=True):
        gt1 = GeneralTools()
        trajectory = []
        start_state = self.markov_model.start_state_index
        end_state = self.markov_model.end_state_index
        previous_step = start_state
        this_step = self.generate_no_gp_step(start_state, 0)
        tra_guidepost_usages = []
        # optimized distribution related step: real end state, revise to non optimization version
        real_end_state = self.choose_end(this_step)
        predicted_length = self.markov_model.calibrator.inner_indices_shortest_path_lengths[this_step, real_end_state]
        multilayer_neighbors = self.get_multilayer_neighbors(real_end_state)
        level1_len_threshold = self.get_level1_threshold_in_use(this_step)
        filtered_time = 0
        level1_step_before = -1
        inner_step_in_this_large_cell = 1
        this_large_cell_inner_trajectory = []
        to_filter = False
        while this_step != end_state:
            trajectory.append(this_step)
            step_number_now = len(trajectory)
            grid = self.markov_model.grid
            level1_step_number = gt1.level1_array_length(trajectory, grid)
            this_step_large_cell = int(grid.level2_subcell_to_large_cell_dict[this_step])
            this_large_cell_dividing_number = grid.level2_subdividing_parameter[this_step_large_cell]
            if level1_step_number != level1_step_before:
                if level1_step_before != -1:
                    to_filter = True
                level1_step_before = level1_step_number
                inner_step_in_this_large_cell = 1
                this_large_cell_inner_trajectory = [this_step]
            else:
                if step_number_now == 2:
                    to_filter = True
                this_large_cell_inner_trajectory.append(this_step)
                inner_step_in_this_large_cell = inner_step_in_this_large_cell + 1
            if this_large_cell_dividing_number > 1:
                inner_this_large_cell_step_ratio = inner_step_in_this_large_cell / this_large_cell_dividing_number
            else:
                inner_this_large_cell_step_ratio = 0
            if to_filter:
                if inner_this_large_cell_step_ratio > 0.4:
                    to_filter = False
                    filtered_time = filtered_time + 1
                    if self.keep_this_trajectory_with_level1_threshold(trajectory, level1_len_threshold,
                                                                       filtered_time) is False:
                        return False
            generating_result = self.end_neighbor_multiplied_next_step(trajectory, this_step, previous_step,
                                                                       step_number_now,
                                                                       level1_step_number, multilayer_neighbors,
                                                                       predicted_length)
            if generating_result is False:
                return False
            this_step = generating_result[0]
            this_step_guidepost_usage = generating_result[1]
            tra_guidepost_usages.append(this_step_guidepost_usage)

            previous_step = trajectory[-1]
            level1_step_number = gt1.level1_array_length(trajectory, grid)
            if level1_step_number <= 2:
                if len(trajectory) > 200:
                    print('this trajectory generation cant stop')
                    return False
            else:
                if len(trajectory) > 100:
                    print('this trajectory generation cant stop')
                    return False
            if len(trajectory) > 8:
                if self.avoid_lingering(np.array(trajectory)):
                    pass
                else:
                    return False
            if neighbor_check:
                if (this_step < end_state - 2) and (previous_step < end_state - 2):
                    neighbor_indicator = self.check_large_neighbor(this_step, previous_step)
                    if neighbor_indicator is True:
                        pass
                    else:
                        return False
        if len(trajectory) == 0:
            return False
        trajectory = np.array(trajectory, dtype=int)
        return trajectory

    def generate_trajectory_without_guidepost(self):
        trajectory = []
        start_state = self.markov_model.start_state_index
        end_state = self.markov_model.end_state_index
        this_step = self.generate_no_gp_step(start_state, 0)
        real_end_state = self.choose_end(this_step)
        multilayer_neighbors = self.get_multilayer_neighbors(real_end_state)
        predicted_length = self.markov_model.calibrator.inner_indices_shortest_path_lengths[
            this_step, real_end_state]
        while (this_step != end_state) and len(trajectory) < 700:
            trajectory.append(this_step)
            step_number_now = len(trajectory)
            this_step = self.no_guidepost_next_step(trajectory, this_step, step_number_now, multilayer_neighbors,
                                                    predicted_length)
        if len(trajectory) == 0:
            # raise ValueError('this trajectory should not be empty')
            return False
        trajectory = np.array(trajectory, dtype=int)
        return trajectory

    def avoid_lingering(self, trajectory: np.ndarray):
        states, frequency_of_states = np.unique(trajectory, return_counts=True)
        large_frequency = - np.sort(- frequency_of_states)
        limited_length = int(np.floor(trajectory.size * self.lingering_if_all_circle_threshold))
        if large_frequency.size > limited_length:
            if np.sum(large_frequency[0:limited_length]) > np.sum(large_frequency) * self.lingering_weight_threshold:
                return False
            else:
                return True
        else:
            return False

    def check_large_neighbor(self, this_step, last_step):
        grid = self.markov_model.grid
        neighbor_relation = grid.large_neighbor_or_same_by_subcell_index(this_step, last_step)
        if neighbor_relation is True:
            return True
        elif neighbor_relation == 'same':
            return True
        else:
            return False

    def end_neighbor_multiplied_next_step(self, trajectory, this_step, previous_step, step_number_now,
                                          level1_step_number, multilayer_neighbors, predicted_length):
        gt1 = GeneralTools()
        cc1 = self.cc
        grid = self.markov_model.grid
        generating_result = self.generate_one_step(this_step, previous_step, step_number_now,
                                                   return_probability=True)
        probability = generating_result[0]
        this_time_guidepost_usage = generating_result[1]
        candidates = np.arange(self.markov_model.noisy_markov_matrix.shape[0])
        if level1_step_number == 1:
            probability[-1] = probability[-1] * 0.5
            if len(trajectory) < predicted_length * 0.5:
                probability[-1] = probability[-1] * 0.2
        probability[-1] = probability[-1] * 0.8
        if np.sum(probability) <= 0:
            neighbors_of_this_step = grid.subcell_neighbors_usable_index[this_step]
            weights = self.total_in_degree[neighbors_of_this_step]
            if np.sum(weights) == 0:
                this_step1 = np.int(gt1.random_pick_element(neighbors_of_this_step))
            else:
                this_step1 = np.int(np.random.choice(neighbors_of_this_step, p=weights / np.sum(weights)))
        else:
            this_step1 = gt1.draw_by_probability_without_an_element(candidates, probability, -2)
        pass
        return this_step1, this_time_guidepost_usage

    #
    def no_guidepost_next_step(self, trajectory, this_step, step_number_now, multilayer_neighbors, predicted_length):
        gt1 = GeneralTools()
        cc1 = self.cc
        probability = self.generate_no_guidepost_one_step(this_step, step_number_now, return_probability=True)
        candidates = np.arange(self.markov_model.noisy_markov_matrix.shape[0])
        if len(trajectory) < predicted_length * 0.5:
            probability[-1] = probability[-1] * 0.2
        this_step1 = gt1.draw_by_probability_without_an_element(candidates, probability, -2)
        return this_step1

    def choose_end(self, start):
        pro = self.markov_model.optimized_start_end_distribution[start, :]
        pro = pro / np.sum(pro)
        gt1 = GeneralTools()
        end = gt1.draw_by_probability(np.arange(pro.size), pro)
        return end

    def generate_many(self, number, neighbor_check=False):
        trajectory_list = []
        if neighbor_check:
            trajectory_number_already = 0
            while trajectory_number_already < number:
                trajectory = self.generate_trajectory(neighbor_check=True)
                if trajectory is not False:
                    trajectory_list.append(trajectory)
                    trajectory_number_already = trajectory_number_already + 1
        else:
            i = 1
            print('begin generating')
            print(datetime.datetime.now())
            while i < number + 1:
                trajectory = self.generate_trajectory()
                if trajectory is not False:
                    trajectory_list.append(trajectory)
                    i = i + 1
            print('end generating')
            print(datetime.datetime.now())
        return trajectory_list
