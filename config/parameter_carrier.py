import numpy as np
# from tools.object_store import ObjectStore


class ParameterCarrier:

    #
    def __init__(self, args):

        self.debug_mode_button = args['debug_mode_button']
        self.dataset_name = args['dataset_name']

        self.total_epsilon = args['total_epsilon']
        self.epsilon_partition = args['epsilon_partition']
        self.level1_divide_inner_parameter = args['level1_divide_inner_parameter']
        self.subdividing_inner_parameter = args['subdividing_inner_parameter']

        self.sensitivity_for_level1_cell_density = args['sensitivity_for_level1_cell_density']

        self.optimization_type = args['optimization_type']


        self.degree_state_distribution_threshold = args['degree_state_distribution_threshold']

        self.trajectory_number_to_generate = args['trajectory_number_to_generate']

        self.fixed_level1_dividing_par = args['fixed_level1_dividing_par']

        self.fixed_level2_divided_large_cell_number = args['fixed_level2_divided_large_cell_number']


