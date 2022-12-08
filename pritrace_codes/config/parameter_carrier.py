class ParameterCarrier:

    def __init__(self, args):
        self.dataset_file_name = args['dataset_file_name']
        self.total_epsilon = args['total_epsilon']
        self.epsilon_partition = args['epsilon_partition']
        self.trajectory_number_to_generate = args['trajectory_number_to_generate']



