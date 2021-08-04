import numpy as np
import os

dataset_name = 'example'

trajectory_data_folder = 'dataset'

cooked_data_name = 'valid_' + dataset_name + '_trajectories'

file_save_folder = 'variables'
visualization_save_directory = 'visualization_result'
trajectory_set_save_name = dataset_name + '_trajectories'

grid_save_name = 'grid'
raw_markov_model_name = 'raw_markov_model'
filtered_model_name = 'filtered_markov_model'
state_trajectories_name = 'sequence_trajectories'
synthetic_gps_trajectories_name = 'synthetic_gps_trajectories'
args_name = 'args'