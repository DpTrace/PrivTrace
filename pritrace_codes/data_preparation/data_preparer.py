from data_preparation.trajectory import Trajectory
from data_preparation.trajectory_set import TrajectorySet
from tools.data_reader import DataReader
from config.parameter_carrier import ParameterCarrier


class DataPreparer:

    def __init__(self, args):
        self.cc = ParameterCarrier(args)

    def get_trajectory_set(self):
        tr_set = TrajectorySet()
        reader1 = DataReader()
        tr_list = reader1.read_trajectories_from_data_file(self.cc.dataset_file_name)
        for tr_array in tr_list:
            tr = Trajectory()
            tr.trajectory_array = tr_array
            tr_set.add_trajectory(tr)
        return tr_set
