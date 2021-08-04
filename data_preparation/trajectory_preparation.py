import numpy as np
import config.folder_and_file_names as config
from data_preparation.data_preparer import DataPreparer
from tools.data_writer import DataWriter
from tools.object_store import ObjectStore
from config.parameter_carrier import ParameterCarrier


class TrajectoryPreparation:

    def __init__(self):
        self.os = ObjectStore()

    def get_trajectory(self, args):
        pc = ParameterCarrier(args)
        data_preparer = DataPreparer(args)
        trajectory_set = data_preparer.set_up_raw_trajectory_set()
        # data_writer = DataWriter()
        # data_writer.save_to_file(trajectory_set, config.trajectory_set_save_name)
        self.os.save_raw_trajectory_set(trajectory_set)

    # a function write trajectory set in folder
