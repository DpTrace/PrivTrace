import numpy as np
import data_cleaning.cleaning_config as config
from tools.math_feature_calculator import MathFeatureCalculator
from config.parameter_carrier import ParameterCarrier
from data_cleaning.raw_data_reader import RawDataReader


class DataCleaner:

    #
    def __init__(self, dataset_name):
        self.feasible_border_north = -1
        self.feasible_border_south = -1
        self.feasible_border_west = -1
        self.feasible_border_east = -1
        self.split_step_length = config.split_step_length
        # self.cc = ConfigureCarrier(args)
        self.mfc = MathFeatureCalculator()
        self.dataset_name = dataset_name
        self.give_cleaner_parameter(dataset_name)

        # parameter I/O
        # this function gives border of grid, order is north, south, west, east

    def give_border(self, value1, direction2):
        if direction2 == 'n':
            self.feasible_border_north = value1
        elif direction2 == 's':
            self.feasible_border_south = value1
        elif direction2 == 'w':
            self.feasible_border_west = value1
        elif direction2 == 'e':
            self.feasible_border_east = value1
        elif direction2 == 'all':
            self.feasible_border_north = value1[0]
            self.feasible_border_south = value1[1]
            self.feasible_border_west = value1[2]
            self.feasible_border_east = value1[3]
        else:
            raise ValueError('wrong direction parameter')

    # this function get border of grid, configure side1 stands for which boder to give. 'n' for north, 's' for south,
    # 'w' for west and 'e' for east. if side1 is 'a', then it means to give out all sides of border in a nd array
    # with the order north, south, west, east

    def get_border(self, direction1):
        if direction1 == 'n':
            border = self.feasible_border_north
        elif direction1 == 's':
            border = self.feasible_border_south
        elif direction1 == 'w':
            border = self.feasible_border_west
        elif direction1 == 'e':
            border = self.feasible_border_east
        elif direction1 == 'all':
            border = np.array([self.feasible_border_north, self.feasible_border_south, self.feasible_border_west,
                               self.feasible_border_east])
        else:
            raise ValueError('wrong side parameter')
        return border

    # this function check if a trajectory is in feasible border
    def if_trajectory_valid(self, data1):
        if_valid_trajectory = 1
        min_latitude = data1[:, 1].min()
        max_latitude = data1[:, 1].max()
        min_longitude = data1[:, 0].min()
        max_longitude = data1[:, 0].max()
        north_border = self.get_border('n')
        south_border = self.get_border('s')
        west_border = self.get_border('w')
        east_border = self.get_border('e')
        if min_latitude < south_border:
            if_valid_trajectory = 0
        if max_latitude > north_border:
            if_valid_trajectory = 0
        if min_longitude < west_border:
            if_valid_trajectory = 0
        if max_longitude > east_border:
            if_valid_trajectory = 0
        return if_valid_trajectory

    # this function delete invalid trajectory array in a trajectory list read
    def discard_invalid_trajectory_array(self, trajectory_array_list):
        invalid_trajectory_index = []
        for trajectory_array_index in range(len(trajectory_array_list)):
            tr_array = trajectory_array_list[trajectory_array_index]
            if tr_array.shape[0] < 1:
                invalid_trajectory_index.append(trajectory_array_index)
            else:
                if_valid_trajectory = self.if_trajectory_valid(tr_array)
                if not if_valid_trajectory:
                    invalid_trajectory_index.append(trajectory_array_index)
        invalid_trajectory_index = np.asarray(invalid_trajectory_index)
        invalid_trajectory_index = np.sort(invalid_trajectory_index)
        for invalid_index in invalid_trajectory_index[::-1]:
            trajectory_array_list.pop(invalid_index)
        return trajectory_array_list

    #
    def get_valid_geolife_data(self, data_range='all'):
        reader1 = RawDataReader()
        directory_list = reader1.geolife_data_directories(data_range)
        valid_array_list = []
        for directory1 in directory_list:
            this_folder_data = self.read_trajectory_in_folder(directory1)
            for trajectory_array1 in this_folder_data:
                dis_car_array = self.get_length_trajectory(trajectory_array1)
                split_result = self.split_trajectories(dis_car_array)
                if split_result is not False:
                    valid_array_list = valid_array_list + split_result
                else:
                    valid_array_list.append(dis_car_array)
        return valid_array_list

    #
    def draw_some_from_whole_dataset(self, tr_list: list, draw_number: int):
        total_number = len(tr_list)
        if draw_number >= total_number:
            return tr_list
        else:
            all_indices = np.arange(total_number)
            np.random.shuffle(all_indices)
            random_indices = all_indices[0: draw_number]
            chosen_trs = []
            for chosen_index in random_indices:
                tr = tr_list[chosen_index]
                chosen_trs.append(tr)
            return chosen_trs

    def get_valid_taxi_data(self, draw_number=-1):
        reader1 = RawDataReader()
        raw_trajectories = reader1.get_taxi_data_from_raw()
        raw_trajectories = self.discard_invalid_trajectory_array(raw_trajectories)
        if draw_number <= 0:
            draw_some = False
        else:
            draw_some = True
        if draw_some:
            tr_list1 = self.draw_some_from_whole_dataset(raw_trajectories, draw_number)
        else:
            tr_list1 = raw_trajectories
        valid_array_list = []
        for trajectory_array1 in tr_list1:
            # trajectory_array1 = directory1
            # for trajectory_array1 in this_folder_data:
            dis_car_array = self.get_length_trajectory(trajectory_array1)
            split_result = self.split_trajectories(dis_car_array)
            if split_result is not False:
                valid_array_list = valid_array_list + split_result
            else:
                valid_array_list.append(dis_car_array)
        return valid_array_list

    #
    def get_valid_roma_data(self):
        reader1 = RawDataReader()
        raw_trajectories = reader1.read_roma_data()
        tr_list1 = self.discard_invalid_trajectory_array(raw_trajectories)
        valid_array_list = []
        for trajectory_array1 in tr_list1:
            dis_car_array = self.get_length_trajectory(trajectory_array1)
            split_result = self.split_trajectories(dis_car_array)
            if split_result is not False:
                valid_array_list = valid_array_list + split_result
            else:
                valid_array_list.append(dis_car_array)
        return valid_array_list

    #
    def get_valid_didi_data(self):
        reader1 = RawDataReader()
        raw_trajectories = reader1.read_didi_data()
        tr_list1 = self.discard_invalid_trajectory_array(raw_trajectories)
        valid_array_list = []
        for trajectory_array1 in tr_list1:
            dis_car_array = self.get_length_trajectory(trajectory_array1)
            split_result = self.split_trajectories(dis_car_array)
            if split_result is not False:
                valid_array_list = valid_array_list + split_result
            else:
                valid_array_list.append(dis_car_array)
        return valid_array_list

    #
    def get_valid_brinkhoff_data(self):
        reader1 = RawDataReader()
        raw_trajectories = reader1.get_brinkhoff_data_from_raw()
        valid_array_list = []
        for trajectory_array1 in raw_trajectories:
            # dis_car_array = self.get_length_trajectory(trajectory_array1)
            dis_car_array = trajectory_array1
            split_result = self.split_trajectories(dis_car_array)
            if split_result is not False:
                valid_array_list = valid_array_list + split_result
            else:
                valid_array_list.append(dis_car_array)
        return valid_array_list

    #
    def get_valid_berlin_data(self):
        latitude_min = self.feasible_border_south
        longitude_min = self.feasible_border_west
        reader1 = RawDataReader()
        raw_trajectories = reader1.read_berlin_data()
        valid_array_list = []
        for trajectory_array1 in raw_trajectories:
            # dis_car_array = self.get_length_trajectory(trajectory_array1)
            trajectory_positive = np.empty(trajectory_array1.shape)
            trajectory_positive[:, 0] = trajectory_array1[:, 0] - longitude_min
            trajectory_positive[:, 1] = trajectory_array1[:, 1] - latitude_min
            dis_car_array = trajectory_positive
            split_result = self.split_trajectories(dis_car_array)
            if split_result is not False:
                valid_array_list = valid_array_list + split_result
            else:
                valid_array_list.append(dis_car_array)
        return valid_array_list

    #
    def get_length_trajectory(self, gps_trajectory: np.ndarray):
        latitude_min = self.feasible_border_south
        longitude_min = self.feasible_border_west
        gps_larger = np.empty(gps_trajectory.shape)
        gps_larger[:, 0] = gps_trajectory[:, 0] - longitude_min
        gps_larger[:, 1] = gps_trajectory[:, 1] - latitude_min
        distance_cartesian_coordinates_array = np.empty(gps_trajectory.shape)
        distance_cartesian_coordinates_array[:, 1] = gps_larger[:, 1] * 111000
        distance_cartesian_coordinates_array[:, 0] =\
            gps_larger[:, 0] * 111000 * (np.cos(gps_trajectory[:, 1] * np.pi / 180))
        return distance_cartesian_coordinates_array

    # find points to split
    def split_trajectories(self, dis_car_array: np.ndarray):
        split_indices = self.split_indices(dis_car_array)
        if split_indices is not False:
            slot_ends_indices = np.unique(np.concatenate(((split_indices + 1), np.array([0, dis_car_array.shape[0]], dtype=int))))
            trajectory_list = []
            for ind_of_indices in range(slot_ends_indices.size - 1):
                start_position_index = slot_ends_indices[ind_of_indices]
                end_position_index = slot_ends_indices[ind_of_indices + 1]
                # if ind_of_indices > 0:
                #     start_position_index = split_indices[ind_of_indices - 1]
                slice_trajectory = dis_car_array[start_position_index:end_position_index, :]
                trajectory_list.append(slice_trajectory)
            return trajectory_list
        else:
            return False

    # this function gives split indices, i.e. split in where data is not continuous(shut down gps device) and staying in
    # a place too long(sleeping)
    def split_indices(self, dis_car_array: np.ndarray):
        step_lengths = self.mfc.calculate_step_lengths(dis_car_array)
        too_long_step_indicator = (step_lengths > self.split_step_length)
        if too_long_step_indicator.any():
            split_indices = np.arange(too_long_step_indicator.size)[too_long_step_indicator]
            return split_indices
        else:
            return False

    # read data
    def read_trajectory_in_folder(self, path1):
        # cleaner1 = DataCleaner()

        reader1 = RawDataReader()
        this_folder_original_data = reader1.read_geolife_data_in_specific_folder(path1)
        trajectory_list1 = self.discard_invalid_trajectory_array(this_folder_original_data)
        return trajectory_list1

    #
    def give_cleaner_parameter(self, dataset_name):

        self.give_border(config.feasible_border_north, 'n')
        self.give_border(config.feasible_border_south, 's')
        self.give_border(config.feasible_border_west, 'w')
        self.give_border(config.feasible_border_east, 'e')

    def get_data(self):
        if self.dataset_name == 'geolife':
            tr_list = self.get_valid_geolife_data()
        elif self.dataset_name == 'taxi':
            tr_list = self.get_valid_taxi_data(200000)
        elif self.dataset_name == 'brinkhoff':
            tr_list = self.get_valid_brinkhoff_data()
        elif self.dataset_name == 'roma':
            tr_list = self.get_valid_roma_data()
        elif self.dataset_name == 'berlin':
            tr_list = self.get_valid_berlin_data()
        elif self.dataset_name == 'didi':
            tr_list = self.get_valid_didi_data()
        return tr_list


