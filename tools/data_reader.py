import numpy as np
import os
import re
import config.folder_and_file_names as config


class DataReader:

    def __init__(self):
        pass

    def read_trajectories_from_data_file(self, file_n):
        file_name = os.path.join('.', config.trajectory_data_folder, file_n)
        trajectory_list = self.read_tra_data(file_name)
        return trajectory_list

    #
    def read_tra_data(self, file_name):
        trajectory_list = []
        f = open(file_name, 'r+')
        for line in f.readlines():
            if line[0] == '>':
                trajectory_data_carrier = line[3:]
                trajectory_data_list = re.split('[,|;*]+', trajectory_data_carrier)[:-1]
                trajectory_data_list = list(map(float, trajectory_data_list))
                trajectory_array = np.array(trajectory_data_list).reshape((-1, 2))
                trajectory_list.append(trajectory_array)
        f.close()
        return trajectory_list
