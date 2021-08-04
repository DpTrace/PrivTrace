import numpy as np
import argparse
import config.folder_and_file_names as fname


class ParSetter:

    #
    def __init__(self):
        pass

    #
    def set_up_args(self, epsilon=False, epsilon_partition=False, level1_parameter=False, level2_parameter=False):
        parser = argparse.ArgumentParser()

        parser.add_argument('--debug_mode_button', type=bool, default=True)

        parser.add_argument('--level1_divide_inner_parameter', type=float, default=600)

        parser.add_argument('--subdividing_inner_parameter', type=float, default=200)

        parser.add_argument('--dataset_name', type=str, default=fname.dataset_name)
        parser.add_argument('--total_epsilon', type=float, default=2.0)
        # regularly, partition solution is suggested to be np.array([0.2, 0.52, 0.28]))
        parser.add_argument('--epsilon_partition', type=np.ndarray, default=np.array([0.2, 0.4, 0.4]))

        parser.add_argument('--sensitivity_for_level1_cell_density', type=float, default=1)

        # 'gravity' or 'default'
        parser.add_argument('--optimization_type', type=str, default='default')

        parser.add_argument('--degree_state_distribution_threshold', type=float, default=5)

        # this parameter indicates how many trajectories to generate
        parser.add_argument('--trajectory_number_to_generate', type=int, default=10000)

        # follows are parameter using in debugging
        # -1 without debugging
        parser.add_argument('--fixed_level1_dividing_par', type=int, default=-1)
        # -1 without debugging
        parser.add_argument('--fixed_level2_divided_large_cell_number', type=int, default=-1)
        args = vars(parser.parse_args())
        if epsilon is not False:
            args['total_epsilon'] = epsilon
        if epsilon_partition is not False:
            args['epsilon_partition'] = epsilon_partition
        if level1_parameter is not False:
            args['level1_divide_inner_parameter'] = level1_parameter
        if level2_parameter is not False:
            args['subdividing_inner_parameter'] = level2_parameter
        return args




