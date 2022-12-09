import numpy as np
import argparse
import config.folder_and_file_names as fname


class ParSetter:

    def __init__(self):
        pass

    def set_up_args(self, dataset_file_name=None, epsilon=False, epsilon_partition=False, level1_parameter=False, level2_parameter=False):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_file_name', type=str, default=fname.dataset_file_name)
        parser.add_argument('--subdividing_inner_parameter', type=float, default=200)
        parser.add_argument('--total_epsilon', type=float, default=2.0)
        # regularly, partition solution is suggested to be np.array([0.2, 0.52, 0.28]))
        parser.add_argument('--epsilon_partition', type=np.ndarray, default=np.array([0.2, 0.4, 0.4]))
        # this parameter indicates how many trajectories to generate
        parser.add_argument('--trajectory_number_to_generate', type=int, default=-1)
        args = vars(parser.parse_args())
        if epsilon is not False:
            args['total_epsilon'] = epsilon
        if epsilon_partition is not False:
            args['epsilon_partition'] = epsilon_partition
        if level1_parameter is not False:
            args['level1_divide_inner_parameter'] = level1_parameter
        if level2_parameter is not False:
            args['subdividing_inner_parameter'] = level2_parameter
        if dataset_file_name is not None:
            args['dataset_file_name'] = dataset_file_name
        return args




