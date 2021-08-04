import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import pickle
from os import walk
import datetime
import config.folder_and_file_names as config
import os
import fcntl


class DataWriter:

    def __init__(self):
        self.files_save_path = os.path.join('.', config.file_save_folder)

    # this function use pickle to save objects not in use
    def save_object(self, object1, path1, file_name1):
        # fw1 = open(path1 + r'\\' + file_name1 + '.txt', 'wb')
        path = os.path.join(path1, file_name1)
        fw1 = open(path, 'wb')
        pickle.dump(object1, fw1)
        fw1.close()

    # this function use pickle to save objects not in use
    def save_syn_trs(self, object1, path1, file_name1):
        # fw1 = open(path1 + r'\\' + file_name1 + '.txt', 'wb')
        path = os.path.join(path1, file_name1)
        fw1 = open(path, 'w')
        pickle.dump(object1, fw1)
        fw1.close()

    # this function is used to save large list in many files
    # def save_objects_in_files(self, object_list1, file_name1):
    #     path1 = self.files_save_path + file_name1
    #     counter1 = 0
    #     for object1 in object_list1:
    #         name1 = str(counter1)
    #         self.save_object(object1, path1, name1)
    #         counter1 += 1

    def save_to_file(self, object1, file_name1, delete_file=True):
        # if isinstance(object, list):
        #     self.save_objects_in_files(object1, file_name1)
        # else:
        #     self.save_object(object1, self.files_save_path, file_name1)
        self.save_object(object1, self.files_save_path, file_name1)
        if delete_file:
            del object1
            gc.collect()

    #
    def save_trajectory_data_in_list_to_file(self, trajectory_list: list, file_path: str):
        file_name = file_path
        with open(file_name, 'w+') as file_object:
            # fcntl.flock(file_object.fileno(), fcntl.LOCK_EX)
            for i in range(len(trajectory_list)):
                file_object.write("#" + str(i)+':\n')
                file_object.write('>0:')
                tr = trajectory_list[i]
                for j in range(tr.shape[0]):
                    x = tr[j, 0]
                    y = tr[j, 1]
                    file_object.write('%.2f' % x + ',' + '%.2f' % y + ';')
                file_object.write('\n')
        # fcntl.flock(file_object.fileno(), fcntl.LOCK_UN)
        file_object.close()


    #
    #
    #
