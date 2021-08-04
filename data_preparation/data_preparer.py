import numpy as np
from data_preparation.trajectory import Trajectory
from data_preparation.trajectory_set import TrajectorySet
from data_cleaning.data_cleaning import DataCleaner
from tools.data_reader import DataReader
from config.parameter_carrier import ParameterCarrier
from tools.math_feature_calculator import MathFeatureCalculator


class DataPreparer:

    #
    def __init__(self, args):
        self.mfc = MathFeatureCalculator()
        self.cc = ParameterCarrier(args)

    #
    def set_up_raw_trajectory_set(self):
        tr_set = TrajectorySet()
        reader1 = DataReader()
        tr_list = reader1.read_trajectories_from_data_file()
        for tr_array in tr_list:
            tr = Trajectory()
            tr.trajectory_array = tr_array
            tr_set.add_trajectory(tr)
        return tr_set

    # #
    # def give_cleaner_parameter(self, cleaner1):
    #     cc1 = self.cc
    #     cleaner1.give_border(cc1.feasible_border_north, 'n')
    #     cleaner1.give_border(cc1.feasible_border_south, 's')
    #     cleaner1.give_border(cc1.feasible_border_west, 'w')
    #     cleaner1.give_border(cc1.feasible_border_east, 'e')

    # def get_valid_geolife_data(self, data_range='all'):
    #     trajectory_set1 = TrajectorySet()
    #     reader1 = DataReader()
    #     directory_list = reader1.geolife_data_directories(data_range)
    #     for directory1 in directory_list:
    #         this_folder_data = self.read_trajectory_in_folder(directory1)
    #         for trajectory_array1 in this_folder_data:
    #             new_trajectory = Trajectory()
    #             new_trajectory.give_trajectory_list(trajectory_array1)
    #             trajectory_set1.add_trajectory(new_trajectory)
    #     return trajectory_set1

    #
    # def get_valid_geolife_data(self, data_range='all'):
    #     trajectory_set1 = TrajectorySet()
    #     reader1 = DataReader()
    #     directory_list = reader1.geolife_data_directories(data_range)
    #     valid_array_list = []
    #     for directory1 in directory_list:
    #         this_folder_data = self.read_trajectory_in_folder(directory1)
    #         for trajectory_array1 in this_folder_data:
    #             dis_car_array = self.get_length_trajectory(trajectory_array1)
    #             split_result = self.split_trajectories(dis_car_array)
    #             if split_result is not False:
    #                 valid_array_list = valid_array_list + split_result
    #             else:
    #                 valid_array_list.append(dis_car_array)
    #     for trajectory_array1 in valid_array_list:
    #         new_trajectory = Trajectory()
    #         new_trajectory.give_trajectory_list(trajectory_array1)
    #         trajectory_set1.add_trajectory(new_trajectory)
    #     return trajectory_set1

    # this function gives split indices, i.e. split in where data is not continuous(shut down gps device) and staying in
    # a place too long(sleeping)
    # def split_indices(self, dis_car_array: np.ndarray):
    #     step_lengths = self.mfc.calculate_step_lengths(dis_car_array)
    #     too_long_step_indicator = (step_lengths > self.cc.split_step_length)
    #     if too_long_step_indicator.any():
    #         split_indices = np.arange(too_long_step_indicator.size)[too_long_step_indicator]
    #         return split_indices
    #     else:
    #         return False

    # find points to split
    # def split_trajectories(self, dis_car_array: np.ndarray):
    #     # step_lengths = self.mfc.calculate_step_lengths(dis_car_array)
    #     # too_long_step_indicator = (step_lengths > self.cc.split_step_length)
    #     # if too_long_step_indicator.any():
    #     #     split_indices = np.arange(too_long_step_indicator.size)[too_long_step_indicator]
    #     split_indices = self.split_indices(dis_car_array)
    #     if split_indices is not False:
    #         slot_ends_indices = np.unique(np.concatenate(((split_indices + 1), np.array([0, dis_car_array.shape[0]], dtype=int))))
    #         trajectory_list = []
    #         for ind_of_indices in range(slot_ends_indices.size - 1):
    #             start_position_index = slot_ends_indices[ind_of_indices]
    #             end_position_index = slot_ends_indices[ind_of_indices + 1]
    #             # if ind_of_indices > 0:
    #             #     start_position_index = split_indices[ind_of_indices - 1]
    #             slice_trajectory = dis_car_array[start_position_index:end_position_index, :]
    #             trajectory_list.append(slice_trajectory)
    #         return trajectory_list
    #     else:
    #         return False

    #
    # def get_length_trajectory(self, gps_trajectory: np.ndarray):
    #     latitude_min = self.cc.feasible_border_south
    #     longitude_min = self.cc.feasible_border_west
    #     gps_larger = np.empty(gps_trajectory.shape)
    #     gps_larger[:, 0] = gps_trajectory[:, 0] - longitude_min
    #     gps_larger[:, 1] = gps_trajectory[:, 1] - latitude_min
    #     distance_cartesian_coordinates_array = np.empty(gps_trajectory.shape)
    #     distance_cartesian_coordinates_array[:, 1] = gps_larger[:, 1] * 111000
    #     distance_cartesian_coordinates_array[:, 0] =\
    #         gps_larger[:, 0] * 111000 * (np.cos(gps_trajectory[:, 1] * np.pi / 180))
    #     return distance_cartesian_coordinates_array




    # # read data
    # def read_trajectory_in_folder(self, path1):
    #     cleaner1 = DataCleaner()
    #     self.give_cleaner_parameter(cleaner1)
    #     reader1 = DataReader()
    #     this_folder_original_data = reader1.read_geolife_data_in_specific_folder(path1)
    #     trajectory_list1 = cleaner1.discard_invalid_trajectory_array(this_folder_original_data)
    #     return trajectory_list1

        # for f, _, i1 in walk(path1):
        #     for j1 in i1:
        #         data = pd.read_csv(f + "\\" + j1, names=['0', '1', '2', '3', '4', '5', '6'], delimiter=',', header=5)
        # if_valid_trajectory = 1
        # min_latitude = data.loc[:, '0'].min()
        # max_latitude = data.loc[:, '0'].max()
        # min_longitude = data.loc[:, '1'].min()
        # max_longitude = data.loc[:, '1'].max()
        # if min_latitude < south_border:
        #     if_valid_trajectory = 0
        # if max_latitude > north_border:
        #     if_valid_trajectory = 0
        # if min_longitude < west_border:
        #     if_valid_trajectory = 0
        # if max_longitude > east_border:
        #     if_valid_trajectory = 0
        # if_valid_trajectory = cleaner1.if_trajectory_valid(data)
        # if if_valid_trajectory:
        #     data.drop(['2', '3', '4', '5', '6'], axis=1, inplace=True)
        #     cols = list(data)
        #     # move the column to head of list using index, pop and insert
        #     cols.insert(0, cols.pop(cols.index('1')))
        #     data = data.loc[:, cols]
        #     temporal_trajectory = Trajectory()
        #     trajectory_list1.append(temporal_trajectory)
        #     trajectory_list1[ind].transform_from_df(data)
        #     trajectory_list1[ind].trajectory_index = outer_ind + ind
        #     ind = ind + 1


    #
    # def read_data(self, path_type='all'):
    #     if path_type == 'all':
    #         trajectory_list1 = self.read_geolife()
    #     else:
    #         path1 = self.root_directory
    #         trajectory_list1 = self.read_trajectory(path1)
    #     return trajectory_list1
    #
    # # special function for reading Geolife dataset
    # def read_geolife(self):
    #     reader1 = DataReader()
    #     trajectory_list11 = []
    #     for i in range(0, 182):
    #         data_path1 = reader1.geolife_file_path(i)
    #         trajectory_list11 += self.read_trajectory(data_path1, len(trajectory_list11))
    #     return trajectory_list11
