from primarkov.mar_model import MarkovModel
from config.parameter_carrier import ParameterCarrier
from data_preparation.trajectory_set import TrajectorySet
from discretization.grid import Grid


class ModelBuilder:

    def __init__(self, cc: ParameterCarrier):
        self.cc = cc

    def build_model(self, grid: Grid, trajectory_set1: TrajectorySet):
        mo1 = MarkovModel(self.cc)
        mo1.model_building(trajectory_set1, grid)
        return mo1

    def filter_model(self, trajectory_set1, grid, mo1):
        mo1.model_filtering(trajectory_set1, grid)
        return mo1