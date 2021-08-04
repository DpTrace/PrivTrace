import numpy as np
from data_preparation.trajectory import Trajectory
from data_preparation.trajectory_set import TrajectorySet
from data_preparation.data_preparer import DataPreparer
from discretization.grid import Grid
from tools.data_reader import DataReader
from tools.data_writer import DataWriter
import config.folder_and_file_names as config
from tools.object_store import ObjectStore
from config.parameter_carrier import ParameterCarrier


class DisData:

    def __init__(self, cc: ParameterCarrier):
        self.os = ObjectStore()
        self.cc = cc
        self.dataset_name = self.cc.dataset_name

    # a function read trajectory set(trajectory_set1) form folder
    # trajectory_set1 = TrajectorySet()
    def get_discrete_data(self):
        # trajectory_file_name = config.trajectory_set_save_name + '.txt'
        # reader = DataReader()
        # trajectory_set1 = reader.read_files_from_path(trajectory_file_name)
        trajectory_set1 = self.os.load_raw_trajectory_set(self.dataset_name)
        grid = Grid(self.cc)
        grid.get_grid(trajectory_set1)
        grid.set_up_state(trajectory_set1)
        trajectory_set1.get_simple_trajectory(grid.real_subcell_index_to_usable_index_dict)
        # data_writer = DataWriter()
        # data_writer.save_to_file(trajectory_set1, config.trajectory_set_save_name)
        # data_writer.save_to_file(grid, config.grid_save_name)


        # self.os.save_raw_trajectory_set(trajectory_set1)
        # self.os.save_grid(grid)

        return trajectory_set1, grid
