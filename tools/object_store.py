import numpy as np
from tools.general_tools import GeneralTools
from primarkov.mar_model import MarkovModel
from primarkov.guidepost import GuidePost
import config.folder_and_file_names as config
from data_preparation.trajectory import Trajectory
from data_preparation.trajectory_set import TrajectorySet
from discretization.grid import Grid
from primarkov.mar_model import MarkovModel
from tools.noise import Noise
from primarkov.sensitive_filter import Filter
from tools.data_reader import DataReader
from tools.data_writer import DataWriter


class ObjectStore:

    #
    def __init__(self):
        self.reader = DataReader()
        self.writer = DataWriter()

    # this function loads grid
    def load_grid(self) -> Grid:
        grid_name = config.grid_save_name
        grid1 = self.reader.read_files_from_path(grid_name)
        return grid1

    # this function loads markov model
    def load_markov_model(self) -> MarkovModel:
        mar_mo_file_name = config.filtered_model_name
        mar_mo = self.reader.read_files_from_path(mar_mo_file_name)
        return mar_mo

    #
    def load_state_trajectories(self) -> list:
        file_name = config.state_trajectories_name
        file = self.reader.read_files_from_path(file_name)
        return file

    # this function save synthetic state trajectories
    def save_state_trajectories(self, tr_list):
        self.writer.save_to_file(tr_list, config.state_trajectories_name)

    # this function save synthetic real trajectories
    def save_synthetic_gps_trajectories(self, tr_list):
        self.writer.save_to_file(tr_list, config.synthetic_gps_trajectories_name)

    # this function save raw trajectories
    def save_raw_trajectory_set(self, trajectory_set):
        self.writer.save_to_file(trajectory_set, config.trajectory_set_save_name)
        # if dataset_name == 'taxi':
        #     self.writer.save_to_file(trajectory_set, config.taxi_trajectory_set_save_name)
        # elif dataset_name == 'geolife':
        #     self.writer.save_to_file(trajectory_set, config.geolife_trajectory_set_save_name)
        # else:
        #     self.writer.save_to_file(trajectory_set, config.trajectory_set_save_name)


    # this function loads raw trajectory set
    def load_raw_trajectory_set(self, dataset_name='default') -> TrajectorySet:
        trajectory_file_name = config.trajectory_set_save_name
        # if dataset_name == 'taxi':
        #     trajectory_file_name = config.taxi_trajectory_set_save_name
        # elif dataset_name == 'geolife':
        #     trajectory_file_name = config.geolife_trajectory_set_save_name
        # else:
        #     trajectory_file_name = config.trajectory_set_save_name
        # trajectory_file_name = config.trajectory_set_save_name
        trajectory_set1 = self.reader.read_files_from_path(trajectory_file_name)
        return trajectory_set1

    # this function saves grid
    def save_grid(self, grid):
        self.writer.save_to_file(grid, config.grid_save_name)

    #
    def load_filtered_markov_model(self) -> MarkovModel:
        file_name = config.filtered_model_name
        file = self.reader.read_files_from_path(file_name)
        return file

    #
    def load_synthetic_gps_trajectories(self) -> list:
        file_name = config.synthetic_gps_trajectories_name
        file = self.reader.read_files_from_path(file_name)
        return file

    # this function saves filtered markov model
    def save_filtered_markov_model(self, file):
        self.writer.save_to_file(file, config.filtered_model_name)

    # this function saves raw markov model
    def save_raw_markov_model(self, file):
        self.writer.save_to_file(file, config.raw_markov_model_name)

    #
    def load_raw_markov_model(self) -> MarkovModel:
        file_name = config.raw_markov_model_name
        file = self.reader.read_files_from_path(file_name)
        return file

    #
    def save_args(self, file):
        self.writer.save_to_file(file, config.args_name)

    #
    def load_args(self):
        file_name = config.args_name
        file = self.reader.read_files_from_path(file_name)
        return file

