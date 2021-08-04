import numpy as np
import os

dataset_name = 'brinkhoff'

raw_geolife_data_root_directory = os.path.join('.', 'dataset', 'Geolife Trajectories 1.3', 'Geolife Trajectories 1.3', 'Data')
raw_taxi_data_root_directory = os.path.join('.', 'dataset', 'train.csv')
raw_brinkhoff_data_root_directory = os.path.join('.', 'dataset', 'brinkhoff_raw.dat')
raw_roma_taxi_data_root_directory = os.path.join('.', 'dataset', 'taxi_february.txt')
raw_berlin_data_root_directory = os.path.join('.', 'dataset', 'berlin_trips.csv')
raw_didi_data_root_directory = file_path = os.path.join('.', 'dataset', 'didi_data', 'chengdu', 'gps_20161001')

largest_data_index_geolife = 181

if dataset_name == 'geolife':
    feasible_border_north = 40.5
    feasible_border_south = 39
    feasible_border_west = 116
    feasible_border_east = 117
elif dataset_name == 'taxi':
    feasible_border_north = 41.4
    feasible_border_south = 41.1
    feasible_border_west = -8.7
    feasible_border_east = -8.4
elif dataset_name == 'brinkhoff':
    feasible_border_north = 31000
    feasible_border_south = 4000
    feasible_border_west = 280
    feasible_border_east = 24000
elif dataset_name == 'roma':
    feasible_border_north = 42.5
    feasible_border_south = 41.5
    feasible_border_west = 12
    feasible_border_east = 13
elif dataset_name == 'berlin':
    feasible_border_north = 27655
    feasible_border_south = -3260
    feasible_border_west = -10625
    feasible_border_east = 33030
elif dataset_name == 'didi':
    feasible_border_north = 30.73
    feasible_border_south = 30.65
    feasible_border_west = 104.04
    feasible_border_east = 104.13

split_step_length = 3500

