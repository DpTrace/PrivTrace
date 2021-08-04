import config.folder_and_file_names as config
from tools.data_reader import DataReader
from tools.data_writer import DataWriter
from primarkov.mar_model import MarkovModel
from tools.object_store import ObjectStore
from config.parameter_carrier import ParameterCarrier
from data_preparation.trajectory_set import TrajectorySet
from discretization.grid import Grid


class ModelBuilder:

    def __init__(self, cc: ParameterCarrier):
        self.cc = cc
        self.os = ObjectStore()

    def build_model(self, grid: Grid, trajectory_set1: TrajectorySet):
        # trajectory_file_name = config.trajectory_set_save_name + '.txt'
        # reader = DataReader()
        # trajectory_set1 = reader.read_files_from_path(trajectory_file_name)
        # grid_file_name = config.grid_save_name + '.txt'
        # grid = reader.read_files_from_path(grid_file_name)
        # trajectory_set1 = self.os.load_raw_trajectory_set()
        # grid = self.os.load_grid()
        mo1 = MarkovModel(self.cc)
        mo1.model_building(trajectory_set1, grid)
        # self.os.save_raw_markov_model(mo1)

        return mo1

        # data_writer = DataWriter()
        # raw_model_name = config.raw_markov_model_name
        # data_writer.save_to_file(mo1, raw_model_name)

    def filter_model(self, trajectory_set1, grid, mo1):
        # trajectory_file_name = config.trajectory_set_save_name + '.txt'
        # reader = DataReader()
        # trajectory_set1 = reader.read_files_from_path(trajectory_file_name)

        # trajectory_set1 = self.os.load_raw_trajectory_set()

        # grid_file_name = config.grid_save_name + '.txt'

        # grid = self.os.load_grid()

        # grid = reader.read_files_from_path(grid_file_name)
        # mo1 = reader.read_files_from_path(config.raw_markov_model_name + '.txt')
        # mo1 = self.os.load_raw_markov_model()

        mo1.model_filtering(trajectory_set1, grid)

        # data_writer = DataWriter()
        # data_writer.save_to_file(mo1, config.filtered_model_name)
        # self.os.save_filtered_markov_model(mo1)
        return mo1