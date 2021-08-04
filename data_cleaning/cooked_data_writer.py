import numpy as np
from tools.data_writer import DataWriter
import pandas as pd
import matplotlib.pyplot as plt
import gc
import pickle
from os import walk
import datetime
import config.folder_and_file_names as config
import os


class CookedDataWriter:

    #
    def __init__(self):
        self.data_save_folder = os.path.join('.', config.trajectory_data_folder)

    #
    def save_cooked_trajectory_data(self, trajectory_list: list, dataset_name='default'):
        file_n = config.cooked_data_name
        file_name = os.path.join(self.data_save_folder, file_n)
        writer = DataWriter()
        writer.save_trajectory_data_in_list_to_file(trajectory_list, file_name)
