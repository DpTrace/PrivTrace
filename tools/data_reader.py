import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import pickle
import os
import re
from os import walk
import datetime
import config.folder_and_file_names as config
from config.parameter_carrier import ParameterCarrier
from data_preparation.trajectory import Trajectory


class DataReader:

    def __init__(self):
        # self.root_directory = root_directory1
        # cc1 = ConfigureCarrier()
        # self.raw_data_root_directory = config.raw_data_root_directory
        self.file_save_directory = config.file_save_folder

        # self.largest_data_index_geolife = cc1.largest_data_index_geolife

    #
    #
    # read

    # this function read object in a file
    def read_from_file(self, file_path1):
        fw = open(file_path1, 'rb')
        trajectory_list1 = pickle.load(fw)
        fw.close()
        return trajectory_list1

    # this function read object processed and saved in specific folder
    def read_files_from_path(self, file_name1):
        # file_path = config.file_save_folder + '/' + file_name1
        file_path = os.path.join('.', config.file_save_folder, file_name1)
        return self.read_from_file(file_path)

    #
    def read_trajectories_from_data_file(self):
        # file_n = config.cooked_data_name_default
        # if dataset_name == 'geolife':
        #     file_n = config.cooked_data_name_geolife
        # if dataset_name == 'taxi':
        #     file_n = config.cooked_data_name_taxi
        file_n = config.cooked_data_name
        file_name = os.path.join('.', config.trajectory_data_folder, file_n)
        trajectory_list = self.read_clean_data(file_name)

        return trajectory_list

    #
    def read_clean_data(self, file_name):
        trajectory_list = []
        f = open(file_name, 'r+')
        for line in f.readlines():
            # if line[0] == '#':
            #     pass
            if line[0] == '>':
                trajectory_data_carrier = line[3:]
                trajectory_data_list = re.split('[,|;*]+', trajectory_data_carrier)[:-1]
                trajectory_data_list = list(map(float, trajectory_data_list))
                trajectory_array = np.array(trajectory_data_list).reshape((-1, 2))
                trajectory_list.append(trajectory_array)
        f.close()
        return trajectory_list



    # this function give full file path of a trajectory in geolife dataset
    # def geolife_file_path(self, index):
    #     root_directory1 = self.raw_data_root_directory
    #     i = index
    #     # if i < 10:
    #     #     ls1 = [root_directory1, '00', i, '/Trajectory']
    #     #     ls2 = [str(i) for i in ls1]
    #     #     ls3 = ''.join(ls2)
    #     # else:
    #     #     if i < 100:
    #     #         ls1 = [root_directory1, '0', i, '/Trajectory']
    #     #         ls2 = [str(i) for i in ls1]
    #     #         ls3 = ''.join(ls2)
    #     #     else:
    #     #         ls1 = [root_directory1, i, '/Trajectory']
    #     #         ls2 = [str(i) for i in ls1]
    #     #         ls3 = ''.join(ls2)
    #     user_string = self.string_for_user_in_directory(i)
    #     # ls1 = [root_directory1, user_string, '/Trajectory']
    #     ls3 = os.path.join(root_directory1, user_string, 'Trajectory')
    #     # ls2 = [str(i) for i in ls1]
    #     # ls3 = ''.join(ls2)
    #     data_path1 = ls3
    #     return data_path1

    #
    # def string_for_user_in_directory(self, index):
    #     if index < 10:
    #         user_index_string = '00' + str(index)
    #     elif index < 100:
    #         user_index_string = '0' + str(index)
    #     else:
    #         user_index_string = str(index)
    #     return user_index_string

    # this function give trajectory to read of geolife dataset. result is a list of strings of directories
    # def geolife_data_directories(self, data_range='all'):
    #     directory_list = []
    #     # this means we want to read all data of geolife
    #     if data_range == 'all':
    #         largest_folder_index = self.largest_data_index_geolife
    #         for dataset_folder_index in range(largest_folder_index + 1):
    #             directory1 = self.geolife_file_path(dataset_folder_index)
    #             directory_list.append(directory1)
    #     elif isinstance(data_range, list) or (type(data_range) is type(np.array([1]))):
    #         start = data_range[0]
    #         end = data_range[-1]
    #         for dataset_folder_index in range(start, end):
    #             directory1 = self.geolife_file_path(dataset_folder_index)
    #             directory_list.append(directory1)
    #     else:
    #         raise TypeError('directory range input is wrong')
    #     return directory_list

    #
    # def read_geolife_data_in_specific_folder(self, path1):
    #     data_list = []
    #     for f, _, i1 in walk(path1):
    #         for j1 in i1:
    #             data_path = os.path.join(f, j1)
    #             # data = pd.read_csv(f + "\\" + j1, names=['0', '1', '2', '3', '4', '5', '6'], delimiter=',', header=5)
    #             data = pd.read_csv(data_path, names=['0', '1', '2', '3', '4', '5', '6'], delimiter=',', header=5)
    #             data.drop(['2', '3', '4', '5', '6'], axis=1, inplace=True)
    #             cols = list(data)
    #             # move the column to head of list using index, pop and insert
    #             cols.insert(0, cols.pop(cols.index('1')))
    #             data = data.loc[:, cols]
    #             trajectory_array = data.values
    #             data_list.append(trajectory_array)
    #     return data_list
