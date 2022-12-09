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
        pass

    def save_trajectory_data_in_list_to_file(self, trajectory_list: list, file_path: str):
        file_name = file_path
        with open(file_name, 'w+') as file_object:
            for i in range(len(trajectory_list)):
                file_object.write("#" + str(i)+':\n')
                file_object.write('>0:')
                tr = trajectory_list[i]
                for j in range(tr.shape[0]):
                    x = tr[j, 0]
                    y = tr[j, 1]
                    file_object.write('%.2f' % x + ',' + '%.2f' % y + ';')
                file_object.write('\n')
        file_object.close()

