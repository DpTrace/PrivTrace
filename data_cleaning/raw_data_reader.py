import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import gc
import pickle
import os
from os import walk
import datetime
import data_cleaning.cleaning_config as config
from config.parameter_carrier import ParameterCarrier
import re


class RawDataReader:

    #
    def __init__(self):
        # self.root_directory = root_directory1
        # cc1 = ConfigureCarrier()
        self.raw_data_root_directory = config.raw_geolife_data_root_directory
        # self.file_save_directory = config.file_save_folder
        self.largest_data_index_geolife = config.largest_data_index_geolife
        # self.cc = ConfigureCarrier()

    #
    # def give_cleaner_parameter(self, cleaner1):
    #     cc1 = self.cc
    #     cleaner1.give_border(cc1.feasible_border_north, 'n')
    #     cleaner1.give_border(cc1.feasible_border_south, 's')
    #     cleaner1.give_border(cc1.feasible_border_west, 'w')
    #     cleaner1.give_border(cc1.feasible_border_east, 'e')

    #
    def string_for_user_in_directory(self, index):
        if index < 10:
            user_index_string = '00' + str(index)
        elif index < 100:
            user_index_string = '0' + str(index)
        else:
            user_index_string = str(index)
        return user_index_string

    # this function give full file path of a trajectory in geolife dataset
    def geolife_file_path(self, index):
        root_directory1 = self.raw_data_root_directory
        i = index
        # if i < 10:
        #     ls1 = [root_directory1, '00', i, '/Trajectory']
        #     ls2 = [str(i) for i in ls1]
        #     ls3 = ''.join(ls2)
        # else:
        #     if i < 100:
        #         ls1 = [root_directory1, '0', i, '/Trajectory']
        #         ls2 = [str(i) for i in ls1]
        #         ls3 = ''.join(ls2)
        #     else:
        #         ls1 = [root_directory1, i, '/Trajectory']
        #         ls2 = [str(i) for i in ls1]
        #         ls3 = ''.join(ls2)
        user_string = self.string_for_user_in_directory(i)
        # ls1 = [root_directory1, user_string, '/Trajectory']
        ls3 = os.path.join(root_directory1, user_string, 'Trajectory')
        # ls2 = [str(i) for i in ls1]
        # ls3 = ''.join(ls2)
        data_path1 = ls3
        return data_path1

    #
    def read_geolife_data_in_specific_folder(self, path1):
        data_list = []
        for f, _, i1 in walk(path1):
            for j1 in i1:
                data_path = os.path.join(f, j1)
                # data = pd.read_csv(f + "\\" + j1, names=['0', '1', '2', '3', '4', '5', '6'], delimiter=',', header=5)
                data = pd.read_csv(data_path, names=['0', '1', '2', '3', '4', '5', '6'], delimiter=',', header=5)
                data.drop(['2', '3', '4', '5', '6'], axis=1, inplace=True)
                cols = list(data)
                # move the column to head of list using index, pop and insert
                cols.insert(0, cols.pop(cols.index('1')))
                data = data.loc[:, cols]
                trajectory_array = data.values
                data_list.append(trajectory_array)
        return data_list

    # this function give trajectory to read of geolife dataset. result is a list of strings of directories
    def geolife_data_directories(self, data_range='all'):
        directory_list = []
        # this means we want to read all data of geolife
        if data_range == 'all':
            largest_folder_index = self.largest_data_index_geolife
            for dataset_folder_index in range(largest_folder_index + 1):
                directory1 = self.geolife_file_path(dataset_folder_index)
                directory_list.append(directory1)
        elif isinstance(data_range, list) or (type(data_range) is type(np.array([1]))):
            start = data_range[0]
            end = data_range[-1]
            for dataset_folder_index in range(start, end):
                directory1 = self.geolife_file_path(dataset_folder_index)
                directory_list.append(directory1)
        else:
            raise TypeError('directory range input is wrong')
        return directory_list

    # def get_array_for_str(self, tr_str, file_index):
    #     path = os.path.join('.', 'taxi_every_trajectory_file_folder')
    #     data_str_list = re.split(r'[\[\],]', tr_str)
    #     data_str_list = list(filter(None, data_str_list))
    #     data_list = list(map(float, data_str_list))
    #     data_list = np.array(data_list)
    #     data_array = data_list.reshape(-1, 2)
    #     file_path = os.path.join(path, str(file_index))
    #     f1 = open(file_path, 'wb+')
    #     pickle.dump(data_array, f1)
    #     f1.close()

    def translate_array_for_str(self, tr_str):
        data_str_list = re.split(r'[\[\],]', tr_str)
        data_str_list = list(filter(None, data_str_list))
        data_list = list(map(float, data_str_list))
        data_list = np.array(data_list)
        data_array = data_list.reshape(-1, 2)
        return data_array

    def get_taxi_data_from_raw(self):
        csv_data = pd.read_csv(config.raw_taxi_data_root_directory)
        trajectory_strs = csv_data.values[:, 8]
        trajectory_list = []
        for i in range(trajectory_strs.size):
            if i % 100 == 0:
                print(i)
            tr_str = trajectory_strs[i]
            data_array = self.translate_array_for_str(tr_str)
            trajectory_list.append(data_array)
        # file_path = os.path.join('.', 'taxi_every_trajectory_file_folder', 'trs')
        # f1 = open(file_path, 'wb+')
        # pickle.dump(trajectory_list, f1)
        # f1.close()
        return trajectory_list

    #
    def get_brinkhoff_data_from_raw(self):
        # meaning of column: 0: state of point, 1: trajectory id, 2: index in trajectory, 3: type of trajectory holder
        # 4: time, 5: x position, 6: y position, 7: speed, 8: next x position, 9: next y position
        tr_list = []
        raw_data = pd.read_csv(config.raw_brinkhoff_data_root_directory, sep='\t', header=None)
        points_data = raw_data.values
        trajectory_number = np.max(points_data[:, 1].astype(int)) + 1
        which_trajectory = points_data[:, 1].astype(int)
        for tr_index in range(trajectory_number):
            in_this_tr = (which_trajectory == tr_index)
            tr_array = points_data[in_this_tr, 5: 7].astype(float)
            tr_list.append(tr_array)
        return tr_list


    #
    def one_id_taxi_trs(self, times, position):
        tr_list = []
        time_intervals = times[1:] - times[:-1]
        interval_seconds = np.empty(time_intervals.size)
        for index in range(time_intervals.size):
            seconds = time_intervals[index].seconds
            interval_seconds[index] = seconds
        different_tr_indicator = (interval_seconds > 2000)
        split_indices = np.arange(different_tr_indicator.size)[different_tr_indicator]
        segment_indices = np.empty((split_indices.size + 1, 2), dtype=int)
        segment_indices[0, 0] = 0
        segment_indices[-1, -1] = position.shape[0]
        for i in range(split_indices.size):
            segment_indices[i, 1] = split_indices[i] + 1
            segment_indices[i + 1, 0] = split_indices[i] + 1
        for j in range(split_indices.size + 1):
            tr_list.append(position[segment_indices[j, 0]: segment_indices[j, 1], :])
        return tr_list

    #
    def read_roma_data(self):
        file_path = config.raw_roma_taxi_data_root_directory
        tr_list = []
        data = pd.read_csv(file_path)
        data1 = data.values
        taxi_ids = np.empty(data1.shape[0], dtype=int)
        times = np.empty(data1.shape[0], dtype=datetime.datetime)
        positions = np.empty((data1.shape[0], 2))
        for index in range(data1.size):
            data_st = data1[index, 0]
            spl_data_sts = data_st.split(';')
            taxi_ids[index] = int(spl_data_sts[0])
            time_str = spl_data_sts[1][:19]
            times[index] = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            position_st = spl_data_sts[2][6: -1]
            position_st = position_st.split(" ")
            positions[index, 0] = float(position_st[0])
            positions[index, 1] = float(position_st[1])
        unids = np.unique(taxi_ids)
        positions = positions[:, [1, 0]]
        for id in unids:
            this_id_taxi_indicator = (taxi_ids == id)
            this_id_times = times[this_id_taxi_indicator]
            this_id_positions = positions[this_id_taxi_indicator, :]
            tr_list = tr_list + self.one_id_taxi_trs(this_id_times, this_id_positions)
        return tr_list

    #
    def read_didi_data(self):
        file_path = config.raw_didi_data_root_directory
        data = pd.read_csv(file_path, header=None)
        data1 = data.values
        unique_order_ids = np.unique(data1[:, 1])
        trips_list = []
        for id_index in range(unique_order_ids.size):
            u1 = unique_order_ids[id_index]
            trip1_indicator = data1[:, 1] == u1
            trip1 = data1[trip1_indicator, 3:5]
            trip1_times = data1[trip1_indicator, 2]
            sorted_trip1 = trip1[np.argsort(trip1_times), :]
            trips_list.append(sorted_trip1)
        return trips_list

    #
    def read_berlin_data(self):
        file_path = config.raw_berlin_data_root_directory
        data = pd.read_csv(file_path)
        data1 = data.values
        trip_ids = np.unique(data1[:, 1])
        trips = []
        for tr_index in range(trip_ids.size):
            this_trip_indicator = (data1[:, 1] == trip_ids[tr_index])
            this_tr_ids = np.arange(data1.shape[0])[this_trip_indicator]
            if this_tr_ids[-1] - this_tr_ids[0] > this_tr_ids.size - 1:
                print('the {}th trip index is wrong'.format(tr_index))
            this_tr = data1[:, 4:6][this_trip_indicator, :]
            if this_tr.shape[0] > 1:
                trips.append(this_tr)
        return trips


