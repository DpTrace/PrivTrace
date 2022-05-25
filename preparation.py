import numpy as np
from data_preparation.trajectory import Trajectory
from data_preparation.trajectory_set import TrajectorySet
from data_preparation.trajectory_preparation import TrajectoryPreparation
from discretization.get_discretization import DisData
from primarkov.build_markov_model import ModelBuilder
from generator.state_trajectory_generation import StateGeneration
from generator.to_real_translator import RealLocationTranslator
from experiment.transition_relationship_error import TransitionError
from experiment.trip_error import TripError
from experiment.length_error import LengthError
from tools.data_reader import DataReader
from data_cleaning.data_cleaning import DataCleaner
from data_cleaning.cooked_data_writer import CookedDataWriter
import config.folder_and_file_names as config
from config.parameter_setter import ParSetter

if __name__ == "__main__":
    args = ParSetter().set_up_args(epsilon=1)
    trapre1 = TrajectoryPreparation()
    trapre1.get_trajectory(args)